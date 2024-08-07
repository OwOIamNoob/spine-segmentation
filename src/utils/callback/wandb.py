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

class WandbCallback(Callback):
    def __init__(self):
        self.images = []
        self.captions = []
        self.table = wandb.Table(
            columns=[
                "Image Name",
                "Slice Index",
                "Image-Channel"
            ]
        )

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger
        print("NOTICEEEEEE!!!!!!!")
        print(type(self.logger))
        self.save_folder = os.path.join(trainer.logger.save_dir, "outputs")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

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

    def visualize(self, predict_seg, img_name = None, save_path = None):
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

        # plt.imshow(image_all)
        # plt.show()
        
        image_all = Image.fromarray(image_all)
        image_all.save(save_path + ".png")

        self.images.append(image_all)
        self.captions.append(img_name)

        # self.logger.log_image(key='Visualize', images=[image_all], caption=[img_name])

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
                        masks={
                            "Pred: Vertebral": {    
                                "mask_data": sample_pred[0, slice_idx, :, :],
                                "class_labels": {1: "Pred: Vertebral"},
                            },
                            "GT: Vertebral": {    
                                "mask_data": sample_label[0, slice_idx, :, :] * 2,
                                "class_labels": {2: "GT: Vertebral"},
                            },

                            "Pred: Canal": {
                                "mask_data": sample_pred[1, slice_idx, :, :] * 3,
                                "class_labels": {3: "Pred: Canal"},
                            },
                            "GT: Canal": {
                                "mask_data": sample_label[1, slice_idx, :, :] * 4,
                                "class_labels": {4: "GT: Canal"},
                            },
                            
                            "Pred: Disk": {
                                "mask_data": sample_pred[2, slice_idx, :, :] * 5,
                                "class_labels": {5: "Pred: Disk"},
                            },
                            "GT: Disk": {
                                "mask_data": sample_label[2, slice_idx, :, :] * 6,
                                "class_labels": {6: "GT: Disk"},
                            },
                        },
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
        prob = outputs["pred"] ## (B, Class, Slice, W, H)
        
        # transforms = monai.transforms.Resize(spatial_size=[60, 280, 280])
        # seg = [monai.transforms.Resize(spatial_size=[60, 280, 280])(volume) for volume in prob] ## Have to do that because monai does not support 4-d affine

        for i in range(len(prob)):
            img_name = batch["image_meta_dict"]["filename_or_obj"][i].split("/")[-1].split(".")[0]
            print("Inference on case {}".format(img_name))

            seg = monai.transforms.Resize(spatial_size=[60, 280, 280])(prob[i])
            self.visualize(seg, img_name, "/work/hpc/spine-segmentation/outputs/images" + img_name)

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

            self.visualize(seg, img_name, "/work/hpc/spine-segmentation/outputs/images" + img_name)
        
            # To save the segmentation volume
            transform = monai.transforms.Resize(spatial_size=[original_size[0], original_size[1], original_size[2]])
            seg = transform(prob[i]) > 0.5
            label = transform(labels[i]) > 0.5
            image = transform(images[i])

            self.log_data_samples_into_tables(image, seg, label, img_name, self.table)

            
            seg = (seg > 0.5).astype(np.int8)
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