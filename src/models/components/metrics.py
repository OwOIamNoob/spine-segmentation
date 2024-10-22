import numpy as np 

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.config import TensorOrList
from monai.utils import convert_data_type, evenly_divisible_all_gather
from monai.metrics import Metric, CumulativeIterationMetric
from torch.nn.modules.loss import _Loss
from lightning.pytorch.loggers import Logger


##### Meter
class Meter(ABC):
    @abstractmethod
    def reset(self):
        pass 

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
    
    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class AverageMeter(Meter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, NdarrayOrTensor):
            self.val = val
            self.sum += val * n                                                                  
            self.count += n
            self.avg = torch.where(self.count > 0, self.sum / self.count, self.sum)
        else: 
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count if self.count > 0 else self.sum
        
    def get(self):
        return self.avg

class IdentityMeter(Meter):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
    
    def update(self, val):
        self.val = val
    
    def get(self):
        if isinstance(self.val, NdarrayOrTensor):
            return self.val.item() if self.val.numel() == 1 else self.val
        return self.val


##### Metric
class NamedMetric(Metric):
    def __init__(self, 
                metric: CumulativeIterationMetric | _Loss | Metric,
                meter: Meter | None = None,
                name: str = "",
                on_step = False):
        super().__init__()
        self.metric = metric
        self.meter = meter
        assert name != "", "Name must not be blank"
        self.name = name
        self.best = 0
    
    @torch.no_grad()
    def __call__(self, pred, gt):
        if isinstance(self.metric, CumulativeIterationMetric):
            self.metric.reset()
            self.metric(pred, gt)
            acc, not_nans = self.metric.aggregate()
            # print(acc)
            self.meter.update(acc, not_nans)
            del acc
            del not_nans
            
        elif isinstance(self.metric, _Loss):
            loss = self.metric(pred, gt)
            loss = loss.item() if loss.numel() == 1 else loss
            self.meter.update(loss, 1)
        
    def log(self, logger, prefix: str, on_step=False, labels=None, addon: str=""):
        # Anouncements hurray
        score = self.meter.get()
        best = 0
        suffix = "_step" if on_step else "_epoch"
        if not np.isscalar(score):
            if isinstance(labels, Sequence):
                # print(labels, score)
                assert len(labels) == len(score), "Metrics do not align with labels {} {}".format(len(labels), len(score))
                for i, label in enumerate(labels): 
                    logger.log("{0}/{1}-{3}{2}".format(label,prefix,self.name, addon) + suffix, 
                                score[i],
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False, 
                                logger=True)

            best = torch.mean(score)
            # Still have to update anyway
            logger.log("{0}/{2}{1}".format(prefix, self.name, addon) + suffix, 
                        best, 
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True, 
                        prog_bar=True, 
                        logger=True)
            
        if np.isscalar(score):
            logger.log("{0}/{2}{1}".format(prefix, self.name, addon) + suffix, 
                            score, 
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True, 
                            prog_bar=True, 
                            logger=True)
            best = score
        
        if best > self.best and not on_step: 
            print("New best {} ({:.6f} --> {:.6f}). ".format(self.name, self.best, best))
            self.best = best
                
    def reset(self):
        self.meter.reset()
    


class MetricCluster(ABC):
    def __init__(self, metrics):
        self.metrics = metrics
        self.logger = None

    def register(self, logger):
        self.logger = logger

    def __call__(self, pred, gt, prefix, labels=None, on_step=False, addon=""):
        for metric in self.metrics:
            metric(pred, gt)
            # When must be log on-step instead of out-range
            if on_step:
                metric.log(self.logger, prefix=prefix, on_step=on_step, labels=labels, addon=addon)
    
    def log(self, prefix, labels=None, addon=""):
        assert self.logger is not None, "No logger to log"
        for metric in self.metrics:
            # No need to log when the metric is constantly logged
            metric.log(self.logger, prefix=prefix, labels=labels, addon=addon)
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()