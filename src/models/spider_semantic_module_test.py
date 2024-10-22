from typing import Any, Dict, Tuple
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.cuda.amp import GradScaler, autocast

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch

from torch_ema import ExponentialMovingAverage as EMA

import numpy as np
import time
import os
import shutil

from functools import partial
import wandb
import inspect 

from src.models.components.metrics import *
from torch.optim.swa_utils import  AveragedModel
from torch_ema import ExponentialMovingAverage as EMA

class SpiderLitModule(LightningModule):
    """

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        sw_batch_size = 4,
        roi = [96, 96, 96],
        infer_overlap = 0.5,
        criterion: torch.nn.modules.loss._Loss = None,
        name=None,
        softmax=True,
        argmax=False,
        degree=2,
        threshold=0.6,
        metric: MetricCluster | None = None,
        ema: EMA | Callable | None = None
    ) -> None:
        """Initialize a `SpiderLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        
        self.ema = ema(net.parameters()) if ema is not None else None

        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=net,
            overlap=infer_overlap,
        )

        self.dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.post_activation = Activations(softmax=True)
        self.post_pred = AsDiscrete(argmax=False, 
                                    threshold=threshold)
        self.degree = degree
        # loss function
        if not criterion:
            self.criterion = DiceLoss(to_onehot_y=False, sigmoid=True, weight = [1, 3, 2])
        else:
            self.criterion = criterion
        
        self.val_criterion = DiceLoss(to_onehot_y=False, 
                                        sigmoid=~softmax, 
                                        softmax=softmax, 
                                        include_background=criterion.include_background)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = AverageMeter()

        if not name:
            self.name = ["None", "Lumbar vertebra", "Spinal canal", "Disk"]
        else: 
            self.name = name  

        # Test with one metric for session
        self.metric = metric
        # The same metric is used for ema parameters for efficiency

        if isinstance(self.metric, MetricCluster):
            self.metric.register(self)
        
    def train_on_device(model):
        super().train_on_device(model)
        if self.ema is not None:
            self.ema.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.train_loss.reset()

    def on_train_epoch_start(self) -> None:
        self.net.train()
    
    def on_train_epoch_end(self) -> None:
        for param in self.net.parameters():
            param.grad = None
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if isinstance(batch, list):
            data, target = batch
        else:
            data, target = batch["image"], batch["label"]
        # print(data.size())
        data, target = data, target
            
        for param in self.net.parameters():
            param.grad = None
            
        with autocast(enabled=False):
            logits = self.net(data)
            # Only take first argument of the function
            loss = self.criterion(logits, target)
        # *Place holder for archived code id 1*
        return loss, logits, target

    # Update EMA
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        if self.ema is not None: 
            self.ema.update()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
    
        loss, logits, targets = self.model_step(batch)
        
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss


    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    
        data, target = batch["image"], batch["label"]
        if isinstance(self.ema, EMA):
            # print("Using Ema")
            with self.ema.average_parameters():
                with autocast(enabled=False):
                    logits = self.model_inferer(data) ## why does it require [b, 4, w, h, d]?????
        else: 
            with autocast(enabled=False):
                logits = self.model_inferer(data)
            
        # Inference
        val_labels_list = decollate_batch(target) ## Optimal to use decollate_batch, we can choose to use it or not
        val_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch, we can choose to use it or not
        val_output_convert = [self.post_pred(self.post_activation(val_pred_tensor * self.degree)) for val_pred_tensor in val_outputs_list]
        
        # Metric computations
        self.metric(val_output_convert, val_labels_list, prefix='val', labels=self.name, on_step=True)
        loss = self.val_criterion(logits, target)
        self.val_loss.update(loss, data.size(0))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'pred': val_output_convert, 'target': target}

    
    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        self.net.eval()
        self.val_loss.reset()
        self.metric.reset()
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        
        self.metric.log('val', labels=self.name)



    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data, target = batch["image"], batch["label"]
        if isinstance(self.ema, EMA):
            # print("Using Ema")
            with self.ema.average_parameters():
                with autocast(enabled=False):
                    logits = self.model_inferer(data) ## logits shape = [B, in_channel, D, W, H]
        else: 
            with autocast(enabled=False):
                logits = self.model_inferer(data) ## logits shape = [B, in_channel, D, W, H]

        test_labels_list = decollate_batch(target) ## Optimal to use decollate_batch, we can choose to use it or not
        test_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch, we can choose to use it or not
        
        test_output_convert = [self.post_pred(self.post_activation(test_pred_tensor)) for test_pred_tensor in test_outputs_list]

        self.metric(test_output_convert, test_labels_list, 'test', labels=self.name, on_step=True)
        # print(self.dice_acc(y_pred=val_output_convert, y=val_labels_list))

        loss = self.val_criterion(logits, target)

        self.val_loss.update(loss, data.size(0))
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'pred': test_output_convert, 'target': target}

    
    def on_test_epoch_start(self,) -> None:
        self.net.eval()
        self.val_loss.reset()
        self.metric.reset()
        pass
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.metric.log('test', labels = self.name)
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss", ##val/loss
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


    def save_checkpoint(self, model, epoch, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
        state_dict = model.state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        filename = os.path.join(self.trainer.log_dir, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)
    


if __name__ == "__main__":
    import rootutils

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    from omegaconf import DictConfig
    import hydra
    print(1)
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        model = hydra.utils.instantiate(cfg.model)
    test()