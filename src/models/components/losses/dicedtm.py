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

# Absolute Module for loss
class Absolute(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.abs(input)

class FPandTNWeight(nn.Module):
    """ Class to map between Gaussian distribution from former Gaussian Blur kernel 
        The degree will affect on the peak of gradient
    """
    def __init__(self, degree = 1):
        super().__init__()
        self.degree = degree

    
    def forward(self, input: Tensor) -> Tensor:
        # Range from -0.6 -> 0.6 ==> (0.5 -> 1.5) * 
        translated = torch.sigmoid(input) * 2 - 1.5
        # Distance weighted and Term weighted (reversedly)
        return torch.exp(translated * self.degree), torch.exp(-translated * self.degree) 

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
        annealing: float = 0.002,
        start_step: int = 10000,
        end_step: int = 50000,
        step: int = 100
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
        self.conv = torch.nn.Sequential(DistanceMapDiceLoss.gradient_window(radius=gradient_kernel, 
                                                                            num_channel=num_classes, 
                                                                            mode=gradient_mode),
                                        Absolute(),
                                        DistanceMapDiceLoss.gaussian_kernel(radius=gaussian_kernel_size,
                                                                            num_channel=num_classes, 
                                                                            delta=gaussian_delta),
                                        FPandTNWeight()).float()
        self.device = "cpu"

        # Class weighting
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

        # Updating gradient map
        self.annealing = annealing
        self.start_step = start_step
        self.end_step = end_step
        self.step = step
        self.global_step = 0

    @classmethod
    def gaussian_kernel(self, radius = 3, num_channel=3, dim=3, delta=0.8):
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
                                padding_mode='zeros'
                                )
        # print(np.repeat(mesh[None, None, :], int(num_channel), axis=0).shape)
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
        # window /= np.sum(window)
        window = - window
        window[r][r][r] = -np.sum(window)
        # Setting up module for forwarding
        gradient = torch.nn.Conv3d( in_channels=int(num_channel), 
                                    out_channels=int(num_channel), 
                                    kernel_size=radius,
                                    groups=int(num_channel),
                                    padding='same', # Keep shape
                                    bias=False,
                                    padding_mode='zeros'
                                    )
        gradient.weight = torch.nn.parameter.Parameter(torch.from_numpy(np.repeat(window[None, None, :], int(num_channel), axis=0)), requires_grad=False)
        return gradient.float()

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
        # Casting layer to input 
        if self.device != input.device:
            self.conv.to(input.device) 
            self.device = input.device

        if self.sigmoid:
            input = torch.sigmoid(input)
  
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

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

        #   Weight computation
        fp_weight, tn_weight = self.conv(target)
        #   Normalize sum to zero
        # print("Weight scale", torch.mean(fp_weight, dim=reduce_axis))
        fp_weight /= torch.mean(fp_weight, dim=reduce_axis)[:, :, None, None, None]
        tn_weight /= torch.mean(tn_weight, dim=reduce_axis)[:, :, None, None, None]
        # print("Distance map weight range:", fp_weight.min(), fp_weight.max())

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                fp_weight = fp_weight[:, 1:]
                tn_weight = tn_weight[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")



        intersection = torch.sum(target * input, dim=reduce_axis)

        denominator = torch.sum(2 * target * input +  fp_weight * (1 - target) * input + tn_weight * target * (1 - input), dim=reduce_axis)
        # print(denominator, 2 * intersection)

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

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
            # apply class_weight to loss
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
        if not export_input and not export_weight:
            return f

        if not export_input:
            input = None
        
        if not export_weight:
            fp_weight = None
        


        return f, input, fp_weight

    def update(self):
        self.global_step += 1
        if self.global_step < self.end_step and self.global_step > self.start_step and (self.global_step - self.start_step) % self.step == 0:
            self.conv[3].degree *= (1 + self.annealing)
    

if __name__ == "__main__":
    import rootutils

    rootutils.setup_root("/work/hpc/spine-segmentation", indicator="setup.py", pythonpath=True)

    from omegaconf import DictConfig
    import hydra
    from copy import deepcopy
    import SimpleITK as sitk
    import monai
    # print(1)

    @hydra.main(version_base="1.3", config_path="../../../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        criterion = DistanceMapDiceLoss(gradient_kernel=7, gaussian_kernel_size=17, gaussian_delta=[4., 8., 8.], num_classes=4)
        # weight = deepcopy(criterion.conv[2].weight).detach().cpu().numpy()
        # img  = sitk.GetImageFromArray(weight)

        dice = monai.losses.DiceLoss()
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        print(type(datamodule))
        loader = iter(datamodule.val_dataloader())
        # batch = next(loader)
        
        # weight = criterion.conv(batch['label'])
        for i in range(3):
            batch = next(loader)
            print("Ref:", dice(torch.softmax(10 * batch['label'] + torch.rand(*batch['label'].shape), dim=1), batch['label']))
            print("Criterion:", criterion(torch.softmax(10 * batch['label'] + torch.rand(*batch['label'].shape), dim=1), batch['label']))
        # label_img = sitk.GetImageFromArray(torch.argmax(batch['label'][1], dim=0).detach().numpy())
        # weight_img = sitk.GetImageFromArray(weight[1, 0].detach().numpy())
        # print(weight.min())
        # print()
        # writer = sitk.ImageFileWriter()
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/weight.nii.gz")
        # writer.Execute(img)
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/sample_weight_25.nii.gz")
        # writer.Execute(weight_img)
        
        # pred = deepcopy(batch["label"])
        # print(pred.dtype)
        # b, c, h, w, d = 2, 3, 32, 280, 280
        # sample = torch.randn(size=[2, 3, 32, 280, 280], fill_value=0.6)
        # print(sample.min(), sample.max(), sample.dtype, sample.device)
        # print(type(sample))
        # gradient, _, _ = criterion(sample, sample)
        # print(gradient.size(), gradient)
        # return datamodule
    
    test()
    