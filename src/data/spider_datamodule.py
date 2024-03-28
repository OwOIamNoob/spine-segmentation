from typing import Any, Dict, Optional

import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold

class SpiderKFoldDataModule(LightningDataModule):
    def __init__(self,
                 data_dir = "./data",
                 json_path = "./data/a.json",
                 transform_train: Optional[monai.transforms.Compose] = None,
                 transform_val: Optional[monai.transforms.Compose] = None,
                 k: int = 5,
                 split_seed: int = 200,
                 num_splits: int = 10, 
                 batch_size: int = 16, 
                 num_workers: int = 8,
                 pin_memory: bool = False,
                ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # data types
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 1 <= self.k <= self.num_splits, "incorrect fold number"
        self.transforms = None
        
        @property
        def num_classes(self):
            return 15
        
        def setup(self, stage: Optional[str] = None):
            if not self.data_train and not self.data_val and not self.data_test:
                dataset_full = SpiderDataset(   data_dir=self.hparams.data_dir,
                                                json_path=self.hparams.json_path)
                
                kf = KFold(n_splits=self.hparams.num_splits, 
                           shuffle=True, 
                           random_state=self.hparams.split_seed)
                train_indexes, val_indexes = [k for k in kf.split(dataset_full)][self.hparams.k]
                self.data_train = dataset_full[train_indexes.tolist()]
                self.data_val = dataset_full[val_indexes.tolist()]
        
        def train_dataloader(self):
            return DataLoader(dataset=self.data_train, 
                              batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.num_workers,
                              pin_memory=self.hparams.pin_memory,
                              shuffle=True)
        def val_dataloader(self):
            return DataLoader(dataset=self.data_train, 
                              batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.num_workers,
                              pin_memory=self.hparams.pin_memory,
                              shuffle=True)
    

        