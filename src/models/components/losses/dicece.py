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
from monai.losses.dice import DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after

### Local module import
import rootutils
# rootutils.setup_root("/work/hpc/spine-segmentation", indicator="setup.py", pythonpath=True)

from src.models.components.losses.w_dice import DistanceMapDiceLoss
from src.utils.weight.spatial import *

class DistanceMapDiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    @deprecated_arg(
        "ce_weight", since="1.2", removed="1.4", new_name="weight", msg_suffix="please use `weight` instead."
    )
    def __init__(
        self,
        dice_dtm: Callable | DistanceMapDiceLoss | DiceLoss,
        softmax: bool = True,
        include_background: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        reduction: str = 'mean',
        batch: bool = False, 
        weight: torch.Tensor | Sequence | None = None,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = reduction

        self.dice = dice_dtm(weight=weight, 
                            reduction=reduction,
                            softmax=softmax,
                            sigmoid=not softmax,
                            batch=batch,
                            include_background=include_background)
        print(type(self.dice))
        self.softmax = softmax
        # Entropy loss will be fused manually. 
        self.include_background = include_background
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none', ignore_index=0 if not include_background else -100, weight=weight)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='none', weight=weight)
        
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

        #DiceLoss configuration
        self.batch = batch
        self.class_weight = weight
        self.device = "cpu"

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        # elif self.old_pt_ver:
        #     warnings.warn(
        #         f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
        #         "Using argmax (as a workaround) to convert target to a single channel."
        #     )
        #     target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target) # type: ignore[no-any-return]

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and target in one single class.

        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.

        """

        if self.device != input.device and self.class_weight is not None:
            self.class_weight = self.class_weight.to(input.device)
            self.cross_entropy.register_buffer("weight", self.class_weight)
            self.binary_cross_entropy.register_buffer("weight", self.class_weight)
            self.device = input.device

        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )
        # Loss forwarding
        # print(input.shape, target.shape)
        dice_loss, weight = self.dice(input, target, export_weight=True)
        # print(weight.shape, input.shape)
        # # For dice loss, the positive sign must be put towards sensitive content
        # print(input.shape)
        # The selected mask of weight       
        # print(torch.mean(input_one_hot.sum(dim=1)), input_one_hot.shape)
        # print(torch.mean(one_hot.sum(dim=1)))
        # print(weight.shape)

        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        if self.softmax:
            ce_loss = self.ce(input, target)
            weight = 1 + torch.zeros_like(target, dtype=torch.float32).scatter_(1, torch.argmax(input, dim=1, keepdim=True), weight)
        else:
            ce_loss = self.bce(input, target)
            weight = 1 + weight
        
        # print(ce_loss.shape, weight.shape)
        # weight = torch.sum(weight, dim=1)
        # print(ce_loss.shape)
        # apply weight

        ce_loss *= weight

        if self.reduction == 'mean':
            ce_loss = torch.mean(ce_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            ce_loss = torch.sum(ce_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(ce_loss.shape[0:2]) + [1] * (len(input.shape) - 2)
            ce_loss = ce_loss.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        
        # print(dice_loss, ce_loss)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
    

if __name__ == "__main__":

    from omegaconf import DictConfig
    import hydra
    from copy import deepcopy
    import SimpleITK as sitk
    import monai
    # print(1)

    @hydra.main(version_base="1.3", config_path="../../../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        # criterion = DistanceMapDiceCELoss(gradient_kernel=7, gaussian_kernel_size=17, gaussian_delta=[4., 8., 8.], num_classes=4, weight=torch.Tensor([0.5, 1., 1., 1.]))
        # weight = deepcopy(criterion.conv[2].weight).detach().cpu().numpy()
        # img  = sitk.GetImageFromArray(weight)

        # dice = monai.losses.DiceLoss()
        # datamodule = hydra.utils.instantiate(cfg.data)
        # datamodule.setup()
        # print(type(datamodule))
        # loader = iter(datamodule.val_dataloader())
        # batch = next(loader)
        # for i in range(500):
        #     criterion.update(20000)
        
        # print(criterion.dice.conv[3])
        # weight = criterion.conv(batch['label'])
        # for i in range(3):
        #     batch = next(loader)
        #     print("Ref:", dice(torch.softmax(10 * batch['label'] + torch.rand(*batch['label'].shape), dim=1), batch['label']))
        #     print("Criterion:", criterion(torch.softmax(10 * batch['label'] + torch.rand(*batch['label'].shape), dim=1), batch['label']))
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
        criterion = hydra.utils.instantiate(cfg.model.criterion)
        # sample = torch.full(size=[2, 4, 32, 256, 256], fill_value=0.6)
        # print(sample.min(), sample.max(), sample.dtype, sample.device)
        # print(type(sample))
        # gradient = criterion(sample.to("cuda:0"), sample.to("cuda:0"))
        # print(gradient.size(), gradient)
        # return datamodule
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        print(type(criterion.dice.spatial_weight))
        print(type(datamodule))
        loader = iter(datamodule.train_dataloader())
        batch = next(loader)
        gradient = criterion(batch['label'].cuda() * 100, batch['label'].cuda())
        print(gradient)
    
    test()
    

