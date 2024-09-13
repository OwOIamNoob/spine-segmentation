import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import lightning as pl
import torch

from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import monai.transforms
import wandb
import nibabel as nib

from src.models.components.losses.dicedtm import DistanceMapDiceLoss
class WandbCallback(Callback):
    def __init__(self, 
                labels, 
                ignore=[], 
                grad=False, 
                radius=3,
                mode='cube', 
                threshold=0.5,
                activation=None):
        self.images = []
        self.captions = []
        self.table = wandb.Table(
            columns=[
                "Image Name",
                "Slice Index",
                "Image-Channel"
            ]
        )
        self.atlas = {}
        for index in range(len(labels)):
            if index in ignore:
                continue
            self.atlas[labels[index]] = index
        self.screen = lambda x, y: 0.2 * x + 0.8 * y
        
        if grad: 
            self.grad = DistanceMapDiceLoss.gradient_window(radius=radius, 
                                                            num_channel=len(labels), 
                                                            mode=mode)
        else:
            self.grad = torch.nn.Identity()
        
        self.device = 'cpu'
        self.post_pred = monai.transforms.AsDiscrete(argmax=False, threshold=threshold)

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger
        print("NOTICEEEEEE!!!!!!!")
        print(type(self.logger))
        self.save_folder = os.path.join(trainer.logger.save_dir, "outputs")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)


    # This function is to merge image and convert it to RGB image.
    def merge_image(self, vertebral_img, disk_img = 0, canal_img = 0, title="", cmap = 'viridis'):
        if isinstance(canal_img, int) == False:
            canal_img = np.stack([canal_img, np.zeros(canal_img.shape), canal_img], axis = -1)
            canal_img = canal_img / canal_img.max() / 2

        if isinstance(disk_img, int) == False:
            disk_img = np.stack([disk_img, disk_img, np.zeros(disk_img.shape)], axis = -1)
            disk_img = disk_img / disk_img.max()

        norm_vertebral = Normalize(vmin=vertebral_img.min(), vmax=vertebral_img.max())

        if cmap == "viridis":
            vertebral_img = cm.viridis(norm_vertebral(vertebral_img.cpu()))[:, :, :3] + disk_img + canal_img
        elif cmap == "inferno":
            vertebral_img = cm.inferno(norm_vertebral(vertebral_img.cpu()))[:, :, :3] + disk_img + canal_img


        # cv2.putText(vertebral_img, title, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1., 0., 0.))

        return vertebral_img

    def  difference(self, pred, gt):
        diff = pred - gt
        fp = (diff > 0).int() * pred 
        tn = (diff < 0).int() * gt
        fp_img = self.build_img(fp, cmap='inferno')
        tn_img = self.build_img(tn, cmap='viridis')
        return fp_img, tn_img
        
        

    def build_img(self, predict_seg, prefix="", cmap='viridis'):
        # coronal_view = zoom(np.sum(predict_seg, axis=2).transpose(0, 2, 1), (1, 1, (predict_seg.shape[2]/predict_seg.shape[1]/6)) ## We need to zoom because the space of images is not the same
        if self.device != predict_seg.device:
            self.grad.to(predict_seg.device)
            self.device = predict_seg.device
        gradient_seg = (torch.abs(self.grad(predict_seg)) > 0.5).to(predict_seg.dtype)
        predict_seg += gradient_seg * predict_seg
        coronal_view = torch.sum(predict_seg, dim=2).permute(0, 2, 1)

        coronal_view = torch.flip(coronal_view,[1]) ## coronal_view[:,::-1,:]

        sagittal_view = torch.sum(predict_seg, dim=1)
        sagittal_view = torch.rot90(sagittal_view, 1, dims=(1, 2))
        sagittal_view = sagittal_view[:,:,80:-50]

        ## coronal_view[0] is the vertebral, coronal_view[1] is the disk, coronal_view[2] is the canal
        ## sagittal_view[0] is the vertebral, sagittal_view[1] is the disk, sagittal_view[2] is the canal

        image1 = self.merge_image(vertebral_img=coronal_view[self.atlas["Vertebrae"]], 
                                    disk_img=coronal_view[self.atlas["Disk"]], 
                                    canal_img=coronal_view[self.atlas["Canal"]], 
                                    title= prefix + "Co", cmap='viridis') ## merge vertebral and canal in coronal view
        image2 = self.merge_image(sagittal_view[self.atlas["Vertebrae"]], 
                                    disk_img=sagittal_view[self.atlas["Disk"]], 
                                    canal_img=sagittal_view[self.atlas["Canal"]], 
                                    title= prefix + "Sa", cmap='viridis') ## merge vertebral, canal and disk in sagittal view
        # Vert
        coronal_view_0 = self.merge_image(coronal_view[self.atlas["Vertebrae"]], 
                                    title= prefix + "Vert", cmap='viridis')
        # Disk 
        coronal_view_2 = self.merge_image(coronal_view[self.atlas["Disk"]], 
                                    title= prefix +"Disk", cmap='viridis')
        # Vert
        sagittal_view_0 = self.merge_image(sagittal_view[self.atlas["Vertebrae"]], 
                                    title= prefix +"Vert", cmap='viridis')
        # Disk
        sagittal_view_2 = self.merge_image(sagittal_view[self.atlas["Disk"]], 
                                    title= prefix +"Disk", cmap='viridis')

        # image2 = np.rot90(image2, 1)

        # print(image1.shape)
        # print(image2.shape)

        image_all = np.concatenate((coronal_view_0, coronal_view_2, image1, sagittal_view_0, sagittal_view_2, image2), 
                                    axis = 1)

        return image_all


    # This function is to get the image in \output (for example /work/hpc/spine-segmentation/outputs/images5_t2.png)
    def visualize(self, predict_seg, img_name = None, save_path = None):
 
        # plt.imshow(image_all)
        # plt.show()
        pred_img = self.build_img(predict_seg)
        
        image_all = (pred_img * 255).astype(np.uint8)
        image_all = Image.fromarray(image_all)
        image_all.save(save_path + ".png")

        self.images.append(image_all)
        self.captions.append(img_name)

        # self.logger.log_image(key='Visualize', images=[image_all], caption=[img_name])
    
    # Abstracted mask building.
    def build_mask(self, pred, label, slice_idx):
        output = {}
        i = 1 
        # Log format for both predictions and labels. 
        #         "Pred: Vertebral": {    
        #     "mask_data": sample_pred[1, slice_idx, :, :],
        #     "class_labels": {1: "Pred: Vertebral"},
        # }
        for title in self.atlas.keys():
            index = self.atlas[title] * 2 - 1
            pred_entry = {f"Pred: {title}" : { "mask_data": pred[self.atlas[title], slice_idx, :, :] * index,
                                                            "class_labels": {index : f"Pred: {title}"}
                                                            }}
            gt_entry =  {f"GT: {title}" : { "mask_data": label[self.atlas[title], slice_idx, :, :] * (index + 1),
                                                            "class_labels": {index + 1 : f"GT: {title}"}
                                                            }}
            output.update(pred_entry)
            output.update(gt_entry)

        return output

    def log_data_samples_into_tables(
        self,
        sample_image: torch.Tensor,
        sample_pred: torch.Tensor,
        sample_label: torch.Tensor,
        image_name: str = None,
        table: wandb.Table = None,
    ):
        print(sample_image.shape)
        sample_image = torch.rot90(sample_image, 1, dims=(2, 3)).cpu().numpy()
        sample_pred = torch.rot90(sample_pred, 1, dims=(2, 3)).cpu().numpy()
        sample_label = torch.rot90(sample_label, 1, dims=(2, 3)).cpu().numpy()

        num_channels, num_slices, _, _ = sample_image.shape
        print(num_slices)
        # with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            ground_truth_wandb_images = []
            for channel_idx in range(num_channels):
                ground_truth_wandb_images.append(
                    wandb.Image(
                        sample_image[channel_idx, slice_idx, :, :],
                        masks = self.build_mask(sample_pred, sample_label, slice_idx),
                    )
                )
            table.add_data(image_name, slice_idx, *ground_truth_wandb_images)
                # progress_bar.update(1)
        return table

    def on_validation_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        images = batch["image"].cuda() ## (B, Channel, Slice, W, H)
        labels = batch["label"].cuda()  ## (B, Class, Slice, W, H)
        # print(batch.keys())
        # print(images.size())
        # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
        # original_size = batch["image_meta_dict"]["spatial_shape"][0]

        # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        # prob = torch.sigmoid(self.model_inferer(image)) ## this or that
        # prob =  ## (B, Class, Slice, W, H)
        
        # transforms = monai.transforms.Resize(spatial_size=[60, 280, 280])
        # seg = [monai.transforms.Resize(spatial_size=[60, 280, 280])(volume) for volume in prob] ## Have to do that because monai does not support 4-d affine

        for i in range(len(outputs['pred'])):
            img_name = batch["image_meta_dict"]["filename_or_obj"][i].split("/")[-1].split(".")[0]
            # print("Inference on case {}".format(img_name))
            pred_seg = monai.transforms.Resize(spatial_size=[60, 280, 280])(outputs["pred"][i])
            target_seg = monai.transforms.Resize(spatial_size=[60, 280, 280])(outputs["target"][i])
            self.visualize(pred_seg, img_name, "/work/hpc/spine-segmentation/outputs/images" + img_name)
            ## To save the segmentation volume
            # seg = monai.transforms.Resize(spatial_size=[original_size[0], original_size[1], original_size[2]])(prob[i])
            
            # seg = (seg > 0.5).astype(np.int8)
            # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            # seg_out[seg[1] == 1] = 2
            # seg_out[seg[0] == 1] = 1
            # seg_out[seg[2] == 1] = 4

            # print(seg_out.shape)
            # nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(self.output_directory, img_name))

    def on_validation_epoch_end(self, trainer, pl_module):
        self.logger.log_image(key='Visualize', images=self.images, caption=self.captions)
        self.images = []
        self.captions = []

    
    def on_test_batch_end(self,
                            trainer: pl.Trainer,
                            pl_module: pl.LightningModule,
                            outputs,
                            batch: Any,
                            batch_idx: int,
                            dataloader_idx: int = 0,
                        ) -> None:
        
        images = batch["image"].cuda() ## (B, Channel, Slice, W, H)
        labels = batch["label"].cuda()  ## (B, Class, Slice, W, H)
        print(batch.keys())
        print(images.size())
        original_size = batch["image_meta_dict"]["spatial_shape"][0]

        # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        # prob = torch.sigmoid(self.model_inferer(image)) ## this or that

        prob = outputs["pred"] ## (B, Class, Slice, W, H)
        
        print(prob[0].shape)
        print(labels[0].shape)
        # transforms = monai.transforms.Resize(spatial_size=[60, 280, 280])

        for i in range(len(prob)):
            img_name = batch["image_meta_dict"]["filename_or_obj"][i].split("/")[-1].split(".")[0]
            original_size = batch["image_meta_dict"]["spatial_shape"][i]

            print("Inference on case {}".format(img_name))

            seg = monai.transforms.Resize(spatial_size=[60, 280, 280])(prob[i]) ## Have to do that because monai does not support 4-d affine

            self.visualize(seg, img_name=img_name, save_path="/work/hpc/spine-segmentation/outputs/images" + img_name)
        
            # To save the segmentation volume
            transform = monai.transforms.Resize(spatial_size=[original_size[0], original_size[1], original_size[2]])
            seg = self.post_pred(transform(prob[i]))
            label = self.post_pred(transform(labels[i]))
            image = transform(images[i])

            self.log_data_samples_into_tables(image, seg, label, img_name, self.table)

            
            seg = seg.astype(np.uint8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4

            print(seg_out.shape) 
            affine = batch["image_meta_dict"]["original_affine"][i].cpu().numpy()

            print(f"Save image at {os.path.join(self.save_folder,img_name)}")

            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(self.save_folder, img_name))

    
    def on_test_epoch_end(self, trainer, pl_module):
        wandb.log({"Test Table": self.table})