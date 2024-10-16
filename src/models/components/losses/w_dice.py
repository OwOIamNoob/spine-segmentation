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

# Disable when training
# import rootutils
# rootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True)


from src.utils.weight.spatial import *

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
        degree: float = 1., 
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReductxion | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: torch.Tensor | None | list = None,
        spatial_weight: nn.Module | torch.Tensor | None = None,
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
        self.degree = degree
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        # Weight properties
        self.norm = norm
        self.spatial_weight = spatial_weight
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor
        
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
                input = torch.softmax(input * self.degree, 1)
            
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
        if isinstance(self.spatial_weight, nn.Module):
            label_weight = self.spatial_weight(target)
            input_weight = self.spatial_weight(input)
            # Forge label and input together
            weight = torch.where((input_weight - label_weight > 0), label_weight, input_weight)
            del label_weight
            del input_weight
        else: 
            # Without spatial weight, it just a normal slower Dice Loss
            weight = torch.zeros_like(input) if self.spatial_weight is None else self.spatial_weight.copy()

        assert weight.shape == target.shape, "Spatial weight must have same weight as prediction"
        # print(weight.shape)
        
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        
        # Computation 
        denominator = torch.sum( input + target +  weight * (input * (1 - target) + (1 - input) * target), dim=reduce_axis)
        numerator = torch.sum(input * target, dim=reduce_axis)
        
        f: torch.Tensor = 1.0 - (2 * numerator + self.smooth_nr) / (denominator + self.smooth_dr)

        # Avoid footprint
        del weight
        del denominator
        del numerator

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

    from omegaconf import DictConfig, OmegaConf
    import hydra
    from copy import deepcopy
    import SimpleITK as sitk
    import monai
    import os
    from src.data.spider_datamodule import *

    # __________________ Additional configuration ________________________________________ #
    OmegaConf.register_new_resolver("zoom", lambda input, ratio: [x * (1 + y) for x, y in zip(input, ratio)])

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
        dice = monai.losses.DiceLoss(sigmoid=False, softmax=False, to_onehot_y=False)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        print(type(criterion.spatial_weight))
        print(type(datamodule))
        loader = iter(datamodule.train_dataloader())
        # label_img = sitk.GetImageFromArray(torch.argmax(batch['label'][1], dim=0).detach().numpy())
       
        # # Always get first sample 
        batch = next(loader)
        print(batch['image'].shape)
        # print(batch['image'].shape, batch['label'].shape)
        # print(torch.argmax(batch['label'][0].detach(), dim=0).shape)
        # tn_weight = criterion.get_weight(batch['label'].to("cuda:3"))
        # print(tn_weight.shape)
        # print(torch.mean(torch.sum(tn_weight, 1), dim=[1, 2, 3]))
        # print("Min and max:", tn_weight.min(), tn_weight.max())

        # writer = sitk.ImageFileWriter()
        
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_input.nii.gz")
        # writer.Execute(sitk.GetImageFromArray(batch['image'][1, 0].detach().numpy()))
        
        # b, c, h, w, d = batch['label'].shape
        # if c == 3:
        #     bg = torch.zeros(b, 1, h, w, d, dtype=batch['label'].dtype, device=batch['label'].device)
        #     batch['label'] = torch.cat([bg, batch['label']], dim=1)
        # # grad =  
        # # print(batch[''])
        # writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_label.nii.gz")
        # writer.Execute(sitk.GetImageFromArray(torch.argmax(batch['label'][0].detach(), dim=0).numpy()))
        
        # for i in range(4):
        #     weight_img = sitk.GetImageFromArray(tn_weight[0, i].detach().cpu().numpy())
            
        #     writer.SetFileName("/work/hpc/spine-segmentation/outputs/dummy/distance_map_{}.nii.gz".format(i))
        #     writer.Execute(weight_img)
        
        # Toy testing 
        criterion.softmax = False
        # criterion.sigmoid = False
        # criterion.magnitude = 0.
        criterion.class_weight = None
        
        refs = []
        scores = []
        for i in range(1):
            refs.append([])
            scores.append([])
            batch = next(loader)
            print(batch["image"].min(), batch["image"].max())
            print(np.unique(batch['label']))
            for j in range(100):
                print("Noise level", 100 - j)
                input, target = (j / 5 + torch.rand(*batch['label'].shape)) * batch['label'], batch['label']
                input = input.to('cuda:3')
                target = target.to('cuda:3')
                print(input.device)
                ref = dice(target, target).detach().cpu().item()
                score = criterion(target, target).detach().cpu().item()
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
    