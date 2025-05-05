from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import rootutils
rootutils.setup_root(search_from=__file__, indicator="setup.py", pythonpath=True)
from src.models.spider_semantic_module_test import SpiderLitModule
import os




path = "/work/hpc/spine-segmentation/logs/train/runs/2024-10-16_19-48-52/checkpoints/last.ckpt"
out_dir = "/work/hpc/spine-segmentation/data/app/checkpoints"
name_net = "dice_dtm.pth"
use_ema = False

with hydra.initialize(version_base="1.3", config_path="../../configs", ):
    cfg = hydra.compose(config_name='train.yaml')
    print(cfg)

# 
# module_checkpoint = torch.load(path)
# print(module_checkpoint.keys())
net = hydra.utils.instantiate(cfg.model.net)
module = SpiderLitModule.load_from_checkpoint(checkpoint_path=path, net=net)
net = module.net
if use_ema :
    try:
        ema_ckpt = module_checkpoint['ema']
        ema_ckpt.copy_to(net)
    except: 
        print("Heheheheee")
# with open(, "w") as file:
torch.save(net.state_dict(), os.path.join(out_dir, name_net))
