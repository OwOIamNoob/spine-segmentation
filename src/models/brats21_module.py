from typing import Any, Dict, Tuple

import torch
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

import numpy as np
import time
import os
import shutil

from functools import partial
import wandb
    
def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

class Brats21LitModule(LightningModule):
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
        roi_x = 96,
        roi_y = 96,
        roi_z = 96,
        infer_overlap = 0.5,
        # amp = False,
    ) -> None:
        """Initialize a `Brast21LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        
        inf_size = [roi_x, roi_y, roi_z]
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=sw_batch_size,
            predictor=net,
            overlap=infer_overlap,
        )
        
        self.dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        
        self.val_acc_max = 0

        # loss function
        self.criterion = DiceLoss(to_onehot_y=False, sigmoid=True)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = AverageMeter()
        self.val_acc = AverageMeter()
        self.test_acc = AverageMeter()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = AverageMeter()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

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
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.test_acc.reset()

    def on_train_epoch_start(self) -> None:
        self.net.train()
        self.train_acc.reset()
    
    def on_train_epoch_end(self) -> None:
        for param in self.net.parameters():
            param.grad = None
        
        # self.train_loss.reset()
        
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # x, y = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # return loss, preds, y
        
        # self.net.train()
        # start_time = time.time()
        # for idx, batch in enumerate(loader):
        if isinstance(batch, list):
            data, target = batch
        else:
            data, target = batch["image"], batch["label"]
        # print(data.size())
        data, target = data.cuda(0), target.cuda(0) #??? vl fix cứng 
            
        for param in self.net.parameters():
            param.grad = None
            
        with autocast(enabled=False):
            logits = self.net(data)
            loss = self.criterion(logits, target)
                

            # loss.backward()
            # optimizer.step()
            
            # self.train_loss.update(loss.item(), n=args.batch_size)
            # if args.rank == 0:
            #     print(
            #         "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            #         "loss: {:.4f}".format(self.train_loss.avg),
            #         "time {:.2f}s".format(time.time() - start_time),
                
        #     start_time = time.time()
        # for param in model.parameters():
        #     param.grad = None
        # return self.train_loss.avg
        
        # with torch.no_grad(): # ????
        #     # for idx, batch in enumerate(loader):
        #     data, target = batch["image"], batch["label"]
        #     data, target = data.cuda(0), target.cuda(0)
        #     with autocast(enabled=False):
        #         logits = self.model_inferer(data)
        #     train_labels_list = decollate_batch(target) ## Optimal to use decollate_batch
        #     train_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch
        #     train_output_convert = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in train_outputs_list]
        #     self.dice_acc.reset()
        #     self.dice_acc(y_pred=train_output_convert, y=train_labels_list)
        #     acc, not_nans = self.dice_acc.aggregate()
        #     acc = acc.cuda(0)

        #     self.train_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
        #     Dice_TC = self.train_acc.avg[0]
        #     Dice_WT = self.train_acc.avg[1]
        #     Dice_ET = self.train_acc.avg[2]

        #     # loss = self.criterion(logits, target)
            
        #     # self.val_loss.update(loss, data.size(0))
        #     # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #     # self.log("val/acc", np.mean(self.val_acc.avg) , on_step=True, on_epoch=True, prog_bar=False, logger=False) ##Mean Val Dice
        #     # self.log("val/acc_best", np.mean(self.val_acc.avg), sync_dist=True, prog_bar=True)
        #     self.log("train/Dice_TC", Dice_TC, on_step=True, on_epoch=True, prog_bar=True)
        #     self.log("train/Dice_WT", Dice_WT, on_step=True, on_epoch=True, prog_bar=True)
        #     self.log("train/Dice_ET", Dice_ET, on_step=True, on_epoch=True, prog_bar=True)
        #     print("Dice_Train_Mean: {:.6f}".format(np.mean(self.train_acc.avg)))
        
        return loss, logits, target

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)
        
        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # loss, logits, targets = self.model_step(batch)

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # self.net.eval()
        # start_time = time.time()
        # run_acc = AverageMeter()

        with torch.no_grad():
            # for idx, batch in enumerate(loader):
            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)
            with autocast(enabled=False):
                logits = self.model_inferer(data)
            val_labels_list = decollate_batch(target) ## Optimal to use decollate_batch, we can choose to use it or not
            val_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch, we can choose to use it or not
            
            # print(logits.shape)
            # print(logits)
    
            # print(type(val_outputs_list))
            # print(len(val_outputs_list))
            val_output_convert = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            # print("++++++++++++++++++++++++++")
            print(val_output_convert)
            self.dice_acc.reset()
            self.dice_acc(y_pred=val_output_convert, y=val_labels_list)
            # print(self.dice_acc(y_pred=val_output_convert, y=val_labels_list))
            acc, not_nans = self.dice_acc.aggregate()
            acc = acc.cuda(0)
            # print("+++++++++")
            # print(acc)

            self.val_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            Dice_TC = self.val_acc.avg[0]
            Dice_WT = self.val_acc.avg[1]
            Dice_ET = self.val_acc.avg[2]

            loss = self.criterion(logits, target)
            
            self.val_loss.update(loss, data.size(0))
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("val/acc", np.mean(self.val_acc.avg) , on_step=True, on_epoch=True, prog_bar=False, logger=False) ##Mean Val Dice
            # self.log("val/acc_best", np.mean(self.val_acc.avg), sync_dist=True, prog_bar=True)
            self.log("val/Dice_TC", Dice_TC, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/Dice_WT", Dice_WT, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/Dice_ET", Dice_ET, on_step=True, on_epoch=True, prog_bar=True)

            # print("Val Dice: Mean: {:.6g}, TC: {:.6f}, WT: {:.6f}, ET: {:.6f}".format(np.mean(self.val_acc.avg), Dice_TC, Dice_WT, Dice_ET))
        # return run_acc.avg
        return {'loss': loss, 'Dice_TC': Dice_TC, 'Dice_WT': Dice_WT, 'Dice_ET': Dice_ET}

    def on_validation_epoch_start(self) -> None:
        self.net.eval()
        # self.val_loss.reset()
        self.val_acc.reset()
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        val_acc = self.val_acc.avg
        semantic_classes = ["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"]
            
        Dice_TC = val_acc[0]
        Dice_WT = val_acc[1]
        Dice_ET = val_acc[2]

        val_avg_acc = np.mean(val_acc)
        self.log("val/acc", val_avg_acc, sync_dist=True, prog_bar=True, logger=False) ##Mean Val Dice
        
        if val_avg_acc > self.val_acc_max:
            print("New best ({:.6f} --> {:.6f}). ".format(self.val_acc_max, val_avg_acc))
            self.val_acc_max = val_avg_acc
            
        # print(val_avg_acc)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
    
    def on_test_epoch_start(self) -> None:
        pass
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
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
    # _ = Brats21LitModule(None, None, None, None)
    print(1)
