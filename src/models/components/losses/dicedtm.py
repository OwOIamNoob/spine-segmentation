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
from torch.nn.modules.loss import _Loss
from torch import Tensor

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after

import FastGeodis as geo 

# For testing performance of weight calculation
import time 

class DistanceMapDiceLoss(_Loss):
    """
    Compute average weighted Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    Weights are initialized based on 
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gradient_kernel: int = 3,
        gradient_mode: str = "cross",
        gaussian_kernel_size: int = 7,
        gaussian_delta: float = 1.5,
        dim: int = 3,
        num_classes: int = 3,
        weight: torch.Tensor | None | list = None,
        threshold: float = 0.5,
        spacing: list[float] = [0.08, 0.04, 0.04],
        global_weight: bool = False,
        gain: float = 0.4,
        inverse: bool = False,
        inverse_background: bool = True,
        norm: bool = False
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        
        # Weight module
        self.gradient = DistanceMapDiceLoss.gradient_window(radius=gradient_kernel, 
                                                            num_channel=num_classes + int(include_background) - 1, 
                                                            mode=gradient_mode)
        self.gaussian = DistanceMapDiceLoss.gaussian_kernel(radius=gaussian_kernel_size,
                                                            num_channel=num_classes + int(include_background) - 1, 
                                                            delta=gaussian_delta)

        # Weight properties
        self.threshold = threshold 
        self.spacing = spacing
        self.global_weight = global_weight
        self.gain = gain
        self.inverse = inverse
        self.inverse_bg = inverse_background

        # Module device
        self.device = "cpu"

        # Class weighting
        self.norm = norm
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor


    @classmethod
    def gaussian_kernel(self, radius=3, num_channel=3, dim=3, delta=0.8):
        # print(radius, type(radius))
        # Define Gaussian kernel
        if isinstance(delta, list) is True:
            delta = np.array(delta)
        

        mean = radius // 2 + ((radius + 1) % 2) / 2 

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
        blur = torch.nn.Conv3d( in_channels=int(num_channel), 
                                out_channels=int(num_channel), 
                                kernel_size=radius,
                                groups=int(num_channel),
                                padding='same', # Keep shape
                                bias=False,
                                padding_mode='reflect'
                                )
        blur.weight = torch.nn.parameter.Parameter(torch.from_numpy(np.repeat(mesh[None, None, :], int(num_channel), axis=0)), requires_grad=False)
        return blur.float()
    
    @classmethod
    def gradient_window(self, radius=3, num_channel=3, mode='cube'):
        """ Constructing window for non-equal supression
        """

        assert radius % 2 != 0
        r = radius // 2
        # add noise to prevent neighbor exclusion
        window = np.ones([radius, radius, radius]) + np.random.randn(radius, radius, radius) * 0.3
        x = np.linspace(-r, r, radius)
        y, z = x.copy(), x.copy()
        xv, yv, zv = np.meshgrid(x, y, z)

        if mode == 'cyclic':
            window[xv**2 + yv**2 + zv**2 >= (r + 0.25) ** 2] = 0
        elif mode == 'cross':
            window *= (xv == 0) | (yv == 0) | (zv == 0)
        elif mode == 'uni-cross':
            x = xv == 0
            y = yv == 0
            z = zv == 0
            window *= (x & y) | (y & z) | (z & x)
        window[r][r][r] = 0
        window = - window
        window[r][r][r] = -np.sum(window)
        # Setting up module for forwarding
        gradient = torch.nn.Conv3d( in_channels=int(num_channel), 
                                    out_channels=int(num_channel), 
                                    kernel_size=radius,
                                    groups=int(num_channel),
                                    padding='same', # Keep shape
                                    bias=False,
                                    padding_mode='reflect'
                                    )
        gradient.weight = torch.nn.parameter.Parameter(torch.from_numpy(np.repeat(window[None, None, :], int(num_channel), axis=0)), requires_grad=False)
        return gradient.float()

    @torch.no_grad()
    def get_weight(self, target: torch.Tensor):
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
        distance_field = distance_field < self.threshold
        if self.global_weight:
            distance_field,_ = torch.min(distance_field, dim=1, keepdim=True)

        spatial_field = deepcopy(target)
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
                distance_weight[i, j] = geo.generalised_geodesic3d(spatial_field[i, j].unsqueeze_(0).unsqueeze_(0),
                                                                    distance_field[i, j].unsqueeze_(0).unsqueeze_(0),  
                                                                    spacing=[0.05, 0.03, 0.03], 
                                                                    v=1, 
                                                                    lamb=0., 
                                                                    iter=4)[0, 0]
                
        # The fact that background is always the negative,
        # Therefore its weight is always on the background
        if self.inverse:
            if self.global_weight or self.inverse_bg or not self.include_background:
                distance_weight = 1 - distance_weight
            else: 
                distance_weight[:, 1:] = 1 - distance_weight[:, 1:] 
        
        # Non-zeros handling 
            distance_weight += self.smooth_dr
        # Handling inference mode
        if self.sigmoid: 
            # if sigmoid, average per-pixel 
            distance_weight /= torch.max(distance_weight, dim=1, keepdim=True)[0]
        # To mitigate weight influence 
        distance_weight = distance_weight * ( 1 - self.gain ) + self.gain

        density = None

        return distance_weight


    def forward(self, input: torch.Tensor, target: torch.Tensor, export_weight=False, export_input=False) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """

        if self.sigmoid:
            input = torch.sigmoid(input)
  
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
            
        # Other act is operated on input
        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        #   Normalize sum to zero
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        #   Weight computation
        weight = self.get_weight(target)
        
        # print(weight.shape)
        
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # error = torch.sum(weight * (target + input - 2 * target * input), dim=reduce_axis)
        
        intersection = input * target
        denominator = torch.sum(input + target, dim=reduce_axis)
        dice_weight = torch.sum(weight * intersection, dim=reduce_axis)
        numerator = torch.sum(intersection, dim=reduce_axis)
        f: torch.Tensor = 1.0 - (2 * numerator + self.smooth_nr) / (denominator + self.smooth_dr)
        w: torch.Tensor = 1.0 - (2 * dice_weight + self.smooth_nr) / (denominator + self.smooth_dr)
        f = f * w
        
        if self.norm:
            f *= (1 / w).detach()
            
        
        # Avoid footprint
        del weight
        del denominator
        del intersection
        del numerator
        del w

        num_of_classes = target.shape[1]

        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss ONLY IF IT WAS NOT NORMED BEFORE
        
            f = f * self.class_weight.to(f)


        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        
        # Return configuration
        return f

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

    @hydra.main(version_base="1.3", config_path="../../../../configs", config_name="train.yaml")
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
        criterion = hydra.utils.instantiate(cfg.model.criterion)
        dice = monai.losses.DiceLoss(sigmoid=False, to_onehot_y=False)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()

        print(type(datamodule))
        loader = iter(datamodule.train_dataloader())
        # label_img = sitk.GetImageFromArray(torch.argmax(batch['label'][1], dim=0).detach().numpy())
       
        # # Always get first sample 
        # batch = next(loader)
        # print(batch['image'].shape, batch['label'].shape)
        # print(torch.argmax(batch['label'][0].detach(), dim=0).shape)
        # tn_weight, norm = criterion.get_weight(batch['label'].to("cuda:3"))
        # print(tn_weight.shape)
        # print(torch.mean(torch.sum(tn_weight, 1), dim=[1, 2, 3]))
        # print(tn_weight.min())
        # print(tn_weight.max())
        # print(norm)
        # writer = sitk.ImageFileWriter()
        
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_input.nii.gz")
        # writer.Execute(sitk.GetImageFromArray(batch['image'][0, 0].detach().numpy()))
        
        # b, c, h, w, d = batch['label'].shape
        # # bg = torch.zeros(b, 1, h, w, d, dtype=batch['label'].dtype, device=batch['label'].device)
        # # batch['label'] = torch.cat([bg, batch['label']], dim=1)
        # # print(batch[''])
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_label.nii.gz")
        # writer.Execute(sitk.GetImageFromArray(torch.argmax(batch['label'][0].detach(), dim=0).numpy()))
        
        # for i in range(3):
        #     weight_img = sitk.GetImageFromArray(tn_weight[0, i].detach().cpu().numpy())
            
        #     writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_{}.nii.gz".format(i))
        #     writer.Execute(weight_img)
        
        # Toy testing 
        criterion.softmax = False
        criterion.sigmoid = False
        criterion.gain = 0.
        criterion.weight = None
        
        refs = []
        scores = []
        for i in range(1):
            refs.append([])
            scores.append([])
            batch = next(loader)
            print(batch["image"].min(), batch["image"].max())
            for j in range(500):
                print("Noise level", 500 - j)
                input, target = torch.softmax( float(j) / 20 * batch['label'] + torch.rand(*batch['label'].shape) * batch['label'], dim=1), batch['label']
                input = input.to('cuda:3')
                target = target.to('cuda:3')
                print(input.device)
                ref = dice(input, target).detach().cpu().item()
                score = criterion(input, target).detach().cpu().item()
                refs[i].append(ref)
                scores[i].append(score)
                print("Ref:", ref)
                print("Criterion:", score)
                np.savetxt("/work/hpc/spine-segmentation/outputs/dummy/dice_loss.txt", refs)
                np.savetxt("/work/hpc/spine-segmentation/outputs/dummy/criterion.txt", scores)

        
        # pred = deepcopy(batch["label"])
        # print(pred.dtype)
        # b, c, h, w, d = 2, 3, 32, 280, 280
        # sample = torch.full(size=[1, 3, 32, 280, 280], fill_value=0.6).to("cuda:2")
        # print(sample.min(), sample.max(), sample.dtype, sample.device)
        # print(type(sample))
        # gradient, _, _ = criterion(sample, sample)
        # print(gradient.size(), gradient)
        # return datamodule
    
    test()
    