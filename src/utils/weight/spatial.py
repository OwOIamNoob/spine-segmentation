# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from monai.utils import Weight, deprecated_arg, look_up_option, pytorch_after

import FastGeodis as geo 

def concave(input, eps=1e-12):
    
    return torch.tanh((input / (1 - input + eps) - 1))

class DistanceMap(nn.Module):
    def __init__(self,
                    gradient_kernel: int = 3,
                    gradient_mode: str = "cross",
                    gaussian_kernel_size: int = 5,
                    gaussian_delta: float = 1.,
                    dim: int = 3,
                    num_classes: int = 4,
                    include_background: bool = True,
                    border_threshold: float = 0.5,
                    spacing: list[float] = [0.08, 0.04, 0.04],
                    global_weight: bool = False,
                    smooth_threshold: float = 0.1,
                    magnitude: float = 0.2,
                    inverse: bool = True,
                    inverse_background: bool = True,
                    post_proc: Callable | None = None) -> None:

        # Weight module
        super().__init__()
        self.gradient = DistanceMap.gradient_window(radius=gradient_kernel, 
                                                    num_channel=num_classes + int(include_background) - 1, 
                                                    dim=dim,
                                                    mode=gradient_mode)
        self.gaussian = DistanceMap.gaussian_kernel(radius=gaussian_kernel_size,
                                                    num_channel=num_classes + int(include_background) - 1, 
                                                    dim=dim,
                                                    delta=gaussian_delta)

        # Weight properties
        self.border_threshold = border_threshold 
        self.spacing = spacing
        self.global_weight = global_weight
        self.magnitude = magnitude
        self.inverse = inverse
        self.inverse_bg = inverse_background
        self.include_background = include_background
        self.smooth_threshold = smooth_threshold
        # Module device
        self.device = "cpu"
        self.post_proc = post_proc
    
    @classmethod
    def default_kernel(self, num_channel, radius, dim):
        # print(mesh.min())
        # Setting up weight for convolution
        if dim == 3:
            kernel = torch.nn.Conv3d( in_channels=int(num_channel), 
                                    out_channels=int(num_channel), 
                                    kernel_size=radius,
                                    groups=int(num_channel),
                                    padding='same', # Keep shape
                                    bias=False,
                                    padding_mode='reflect'
                                    )
        elif dim == 2:
            kernel = torch.nn.Conv2d( in_channels=int(num_channel), 
                                    out_channels=int(num_channel), 
                                    kernel_size=radius,
                                    groups=int(num_channel),
                                    padding='same', # Keep shape
                                    bias=False,
                                    padding_mode='reflect'
                                    )
        else:
            kernel = torch.nn.Conv1d( in_channels=int(num_channel), 
                                    out_channels=int(num_channel), 
                                    kernel_size=radius,
                                    groups=int(num_channel),
                                    padding='same', # Keep shape
                                    bias=False,
                                    padding_mode='reflect'
                                    )
        
        return kernel

    @classmethod
    def gaussian_kernel(self, radius=3, num_channel=3, dim=3, delta=0.8):
        # print(radius, type(radius))
        # Define Gaussian kernel
        if isinstance(delta, list) is True:
            delta = np.array(delta)
        

        mean = radius // 2 + ((radius + 1) % 2) / 2 
        print(dim)
        coef = np.identity(dim) * np.power(delta, 2)
        print(coef)
        inv_coef = np.linalg.inv(coef)

        denominator = (((2 * np.pi) ** dim) * np.linalg.det(coef)) ** 0.5
        print(denominator)
        grid = np.arange(radius)
        mesh = np.array(np.meshgrid(*[grid] * dim)) - mean
        mesh = mesh.astype(np.float64)
        mesh = np.apply_along_axis(lambda x: np.exp( - 0.5 * x.T @ inv_coef @ x ), 0, mesh) / denominator
        mesh = np.abs(mesh)
        mesh /= np.sum(mesh)
        # print(mesh.min())
        # Setting up weight for convolution
        blur = self.default_kernel(num_channel, radius, dim)
        blur.weight = torch.nn.parameter.Parameter(torch.from_numpy(np.repeat(mesh[None, None, :], int(num_channel), axis=0)), requires_grad=False)
        return blur.float()
    
    @classmethod
    def gradient_window(self, radius=3, dim=3, num_channel=3, mode='cube'):
        """ Constructing window for non-equal supression
        """

        assert radius % 2 != 0
        r = radius // 2
        # add noise to prevent neighbor exclusion
        size = [radius] * dim
        window = np.ones(size) + np.random.randn(*size) * 0.3
        x = np.linspace(-r, r, radius)
        ax = np.meshgrid(*[x] * dim)

        if mode == 'cyclic':
            distance = np.stack(ax, axis=-1)
            distance = np.sum(distance ** 2, axis=-1)
            window[distance >= (r + 0.25) ** 2] = 0
        elif mode == 'cross':
            valid = np.stack(ax, axis=-1)
            print(np.max(valid == 0, axis=-1))
            window *=  np.max(valid == 0, axis=-1)
        if dim == 2:
            window[r, r] = 0
            window[r, r] = -np.sum(window)
        elif dim == 3:
            window[r, r, r] = 0
            window[r, r, r] = -np.sum(window)
        window = - window
        
        print(window)
        gradient = self.default_kernel(num_channel, radius, dim)
        gradient.weight = torch.nn.parameter.Parameter(torch.from_numpy(np.repeat(window[None, None, :], int(num_channel), axis=0)), requires_grad=False)
        return gradient.float()

    @torch.no_grad()
    def forward(self, target: torch.Tensor):
        # FastGeodis hasn't support batch inference yet, we need to de-batch and re-batch :) \
        # And I figured out that it also not support multi-channel, so hell.
        # If using global weight, only the background weight is calculated since it the negative merge of all indexes.
        if self.device != target.device:
            self.gaussian.to(target.device)
            self.gradient.to(target.device)
            self.device = target.device
        # current = time.time()
        # Pass and blur the gradient.
        distance_field = self.gaussian(torch.sigmoid(self.gradient(target)))
        # print(distance_field.max(), distance_field.min())
        # Inverse mask to calculate distance field to object
        if self.global_weight:
            distance_field,_ = torch.max(distance_field, dim=1, keepdim=True)
        distance_field = distance_field < self.border_threshold
        
        spatial_field = torch.clone(target)
        # Take inverse image
        if self.include_background:
            #   We don't need to inverse distance map 
            #   since it derived gradient values from original image
            if self.global_weight:
                spatial_field = 1 - spatial_field[:, 0].unsqueeze_(1)
            else:
                if not self.inverse_bg:
                    spatial_field[:, 1:] = 1 - spatial_field[:, 1:] 
                else:
                    spatial_field = 1 - spatial_field
        else: 
            if self.global_weight:
                spatial_field, _ = 1 - torch.max(spatial_field, dim=1, keepdim=True)
            spatial_field = 1 - spatial_field

        distance_weight = torch.ones_like(spatial_field)
        # Follows current object size for synchronization
        for i in range(spatial_field.shape[0]):
            for j in range(spatial_field.shape[1]):
                if len(spatial_field.shape) == 5:
                    distance_weight[i, j] = geo.generalised_geodesic3d(spatial_field[i, j].unsqueeze_(0).unsqueeze_(0),
                                                                        distance_field[i, j].unsqueeze_(0).unsqueeze_(0),  
                                                                        spacing=self.spacing, 
                                                                        v=1, 
                                                                        lamb=0., 
                                                                        iter=4)[0, 0]
                elif len(spatial_field.shape) == 4:
                    distance_weight[i, j] = geo.generalised_geodesic3d(spatial_field[i, j].unsqueeze_(0).unsqueeze_(0).unsqueeze_(0),
                                                                        distance_field[i, j].unsqueeze_(0).unsqueeze_(0).unsqueeze_(0),
                                                                        spacing=self.spacing,
                                                                        v=1, 
                                                                        lamb=0., 
                                                                        iter=4)[0, 0, 0]
                
        # The fact that background is always the negative,
        # Therefore its weight is always on the background
        # distance_weight /= torch.amax(distance_weight, dim=[2, 3, 4])[:, :, None, None, None]
        # The smooth threshold will decide the trade-off between smooth and stimuli
        distance_weight -= self.smooth_threshold

        # If not inversed, the background will be smoothed, otherwise, the foreground
        if self.inverse:
                distance_weight = -distance_weight 
        
        if self.inverse_bg and distance_weight.size(1) > 1:
                distance_weight[:, 0] = -distance_weight[:, 0] 
        # print(distance_weight.max(), distance_weight.min())
        if self.post_proc is not None:
            # print("Proc")
            distance_weight =  self.post_proc(distance_weight)
        return distance_weight * self.magnitude


if __name__ == "__main__":
    import rootutils

    rootutils.setup_root("/work/hpc/spine-segmentation", indicator=".project-root", pythonpath=True)

    from omegaconf import DictConfig
    import hydra
    from copy import deepcopy
    import SimpleITK as sitk
    import monai
    import os
    from src.data.spider_datamodule import *
    from functools import partial 
    import cv2

    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        # criterion = DistanceMapDiceLoss(gradient_kernel=7, 
        #                                 gaussian_kernel_size=11, 
        #                                 gaussian_delta=[2., 5., 5.], 
        #                                 num_classes=4, 
        #                                 include_background=True,
        #                                 global_weight=False,
        #                                 gain=0.5,
        #                                 inverse=True)
        # dice = DiceLoss()
        # weight = deepcopy(criterion.conv[2].weight).detach().cpu().numpy()
        # img  = sitk.GetImageFromArray(weight)
        # criterion = hydra.utils.instantiate(cfg.model.criterion)
        # dice = monai.losses.DiceLoss(sigmoid=False, to_onehot_y=False)
        # proc = partial(torch.nn.functional.normalize, p=2.0, dim=[2, 3, 4], eps=1e-12)
        dtm = DistanceMap(  post_proc=None, 
                            inverse=False,
                            num_classes=4,
                            gaussian_kernel_size=13,
                            gaussian_delta=2.,
                            dim=3,
                            )
        # radius = 10 
        # r = 5
        # a = np.linspace(-radius, radius, 150)
        # b = np.stack(np.meshgrid(a, a), axis=-1)
        # b = np.sum(b ** 2, axis=2)
        # c = np.zeros_like(b)
        # c[b <= r**2] = 1
        # print(c.shape)
        # c = torch.from_numpy(c).cuda()[None, None, :, :].to(dtype=torch.float32)
        # print(c.shape, "???")
        # dstm = dtm(c).detach().cpu().numpy()[0, 0]
        # dstm -= dstm.min()
        # dstm = (dstm * 255).astype(np.uint8)
        # dstm = np.stack([dstm, dstm, dstm], axis=-1) 
        # print(dstm.shape)
        # cv2.imwrite("/work/hpc/spine-segmentation/src/utils/weight/2d_geo.jpg", dstm)
        
        # test = torch.randn([1, 1, 512, 512]) > 0.4
        # test = test.view(torch.float32).cuda()
        # dtm = dtm(test)
        # print(dtm.post_proc is None)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()


        print(type(datamodule))
        loader = iter(datamodule.val_dataloader())
        # label_img = sitk.GetImageFromArray(torch.argmax(batch['label'][1], dim=0).detach().numpy())
       
        # # Always get first sample 
        batch = next(loader)
        tn_weight = dtm(batch['label'].to('cuda:0'))
        print(tn_weight.shape)
        # print(torch.mean(torch.sum(tn_weight, 1), dim=[1, 2, 3]))
        print("Min and max:", tn_weight.min(), tn_weight.max())

        writer = sitk.ImageFileWriter()
        
        writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_input.nii.gz")
        writer.Execute(sitk.GetImageFromArray(batch['image'][0, 0].detach().numpy()))
        
        b, c, h, w, d = batch['label'].shape
        if c == 3:
            bg = torch.zeros(b, 1, h, w, d, dtype=batch['label'].dtype, device=batch['label'].device)
            batch['label'] = torch.cat([bg, batch['label']], dim=1)
        bg = batch['label'][:, [0]]
        print(torch.sum(bg * batch['label'][:, 1:], dim=[0, 2, 3, 4]), bg.unique())
        
        # grad =  
        # print(batch[''])
        writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_label.nii.gz")
        writer.Execute(sitk.GetImageFromArray(torch.argmax(batch['label'][0].detach(), dim=0).numpy()))
        # writer.Execute(sitk.GetImageFromArray(batch['label'][0, 0].detach().numpy()))
        
        for i in range(4):
            weight_img = sitk.GetImageFromArray(tn_weight[0, i].detach().cpu().numpy())
            
            writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_{}.nii.gz".format(i))
            writer.Execute(weight_img)
        
    test()

