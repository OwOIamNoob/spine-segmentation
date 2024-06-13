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
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

class SpiderCallback(Callback):
    def __init__(self, 
                 log_size: 10,
                 volume: [96, 96, 96]):
        self.volume = volume
    
    def on_train_start(self,
                       trainer: pl.Trainer,
                       pl_module: pl.LightningModule):
        logger = trainer.logger
        logger.