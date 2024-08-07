import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from functools import partial

import nibabel as nib
import numpy as np

from lightning import LightningModule
import torch
from torch.utils.data import DataLoader, Dataset
# from src.models.brats21_module import Brats21LitModule
from src.models.spider_semantic_module import SpiderLitModule   
from monai.inferers import sliding_window_inference

from monai.transforms.utils import allow_missing_keys_mode
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

import monai.transforms

class Inferer(object):
    def __init__(
        self,
        exp_name: str,
        checkpoint_path: str,
        data_name: str,
        data_loader: DataLoader,
        net: torch.nn.Module,
        model_inferer: sliding_window_inference,
        device
    ):
        self.output_directory = "/work/hpc/spine-segmentation/outputs/" + exp_name ##"./outputs/" + exp_name
        print(os.getcwd())
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        self.data_name = data_name
        self.data_loader = data_loader
        net.to(device)
        self.net = net
        model = SpiderLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.model_inferer = partial(model_inferer, predictor=model)
        

    def feed(self):
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                image = batch["image"].cuda()
                label = batch["label"].cuda()
                print(batch.keys())
                print(image.size())
                affine = batch["image_meta_dict"]["original_affine"][0].numpy()
                original_size = batch["image_meta_dict"]["spatial_shape"][0]

                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                # plt.imshow(label[0][0][10].cpu())
                prob = torch.sigmoid(self.model_inferer(image)) ## this or that
                # prob = torch.sigmoid(self.net(image)) ## that

                # seg = prob[0].detach().cpu().numpy()

                print(prob.shape)

                # transforms = monai.transforms.Resize(spatial_size=[60, 280, 280])
                seg = [monai.transforms.Resize(spatial_size=[60, 280, 280])(volume) for volume in prob] ## Have to do that because monai does not support 4-d affine

                self.logger(seg[0], os.path.join(self.output_directory, "images", img_name.split(".")[0]))

                # seg = zoom(seg, (1, float(original_size[0])/seg.shape[1], float(original_size[1])/seg.shape[2], float(original_size[2])/seg.shape[3]))
                seg = [monai.transforms.Resize(spatial_size=[original_size[0], original_size[1], original_size[2]])(volume) for volume in prob]
                
                seg = seg[0]

                print(seg.shape)
                seg = (seg > 0.5).astype(np.int8)
                print("Bruhhhh")
                print(seg.shape)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4

                print(seg_out.shape)
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(self.output_directory, img_name))
            print("Finished inference!")
        
        # for i, batch in enumerate(self.data_loader):
        #     print(batch['image'].size())
        #     if (i == 0):
        #         print(batch['image'][0])

    def merge_image(self, vertebral_img, disk_img = 0, canal_img = 0):
        if isinstance(canal_img, int) == False:
            canal_img = np.stack([canal_img, np.zeros(canal_img.shape), canal_img], axis = -1)
            canal_img = canal_img / canal_img.max() / 2

        if isinstance(disk_img, int) == False:
            disk_img = np.stack([disk_img, disk_img, np.zeros(disk_img.shape)], axis = -1)
            disk_img = disk_img / disk_img.max()

        norm_vertebral = Normalize(vmin=vertebral_img.min(), vmax=vertebral_img.max())

        vertebral_img = cm.viridis(norm_vertebral(vertebral_img.cpu()))[:, :, :3] + disk_img + canal_img

        return vertebral_img

    def logger(self, predict_seg, save_path = None):
        # coronal_view = zoom(np.sum(predict_seg, axis=2).transpose(0, 2, 1), (1, 1, (predict_seg.shape[2]/predict_seg.shape[1]/6)) ## We need to zoom because the space of images is not the same
        coronal_view = torch.sum(predict_seg, dim=2).permute(0, 2, 1)

        coronal_view = torch.flip(coronal_view,[1]) ## coronal_view[:,::-1,:]

        sagittal_view = torch.sum(predict_seg, dim=1)
        sagittal_view = torch.rot90(sagittal_view, 1, dims=(1, 2))
        sagittal_view = sagittal_view[:,:,80:-50]

        ## coronal_view[0] is the vertebral, coronal_view[1] is the disk, coronal_view[2] is the canal
        ## sagittal_view[0] is the vertebral, sagittal_view[1] is the disk, sagittal_view[2] is the canal

        image1 = self.merge_image(coronal_view[0], coronal_view[2])
        image2 = self.merge_image(sagittal_view[0], sagittal_view[2], sagittal_view[1])
        coronal_view_0 = self.merge_image(coronal_view[0])
        coronal_view_2 = self.merge_image(coronal_view[2])
        sagittal_view_0 = self.merge_image(sagittal_view[0])
        sagittal_view_2 = self.merge_image(sagittal_view[2])

        # image2 = np.rot90(image2, 1)

        print(image1.shape)
        print(image2.shape)

        image_all = np.concatenate((coronal_view_0, coronal_view_2, image1, sagittal_view_0, sagittal_view_2, image2), axis = 1)
        image_all = (image_all * 255).astype(np.uint8)

        plt.imshow(image_all)
        plt.show()
        
        image_all = Image.fromarray(image_all)
        image_all.save(save_path + ".png")
        
            
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf, DictConfig
    
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs/app")
    
    @hydra.main(version_base="1.3", config_path=config_path, config_name="infer_spider")
    def main(cfg: DictConfig):
        # print(OmegaConf.to_yaml(cfg))
        inferer: Inferer = hydra.utils.instantiate(cfg)
        print(inferer)
        # print(type(inferer))
        inferer.feed()
        
    main()
