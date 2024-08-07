from typing import Any, Dict, Optional

import torch
import monai
from monai import transforms
from monai.data.utils import list_data_collate, pad_list_data_collate
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold
import rootutils

rootutils.setup_root(search_from=__file__, indicator="setup.py", pythonpath=True)
from src.data.components.spider_dataset import SpiderDataset, SpiderTransformedDataset

class SpiderKFoldDataModule(LightningDataModule):
    def __init__(self,
                 data_dir = "./data",
                 json_path = "./data/a.json",
                 spacing = [1., 1., 1.],
                 transform_train: Optional[monai.transforms.Compose] = None,
                 transform_val: Optional[monai.transforms.Compose] = None,
                 k: int = 5, # no. epoch before switching fold
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
        assert 1 <= self.hparams.k <= self.hparams.num_splits, "incorrect fold number"

    
    
    @property
    def num_classes(self):
        return 4
    
    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_full = SpiderDataset(   data_dir=self.hparams.data_dir,
                                            json_path=self.hparams.json_path)
            
            kf = KFold(n_splits=self.hparams.num_splits, 
                        shuffle=True, 
                        random_state=self.hparams.split_seed)
            train_indexes, val_indexes = [k for k in kf.split(dataset_full)][self.hparams.k]
            self.data_train = SpiderDataset(data = dataset_full[train_indexes.tolist()])
            self.data_val = SpiderDataset(data = dataset_full[val_indexes.tolist()])
    
    def get_transformed_dataset(self, dataset, transform):
        return SpiderTransformedDataset(dataset, transform)
    
    def train_dataloader(self) -> DataLoader[Any]: 
        return DataLoader(dataset=self.get_transformed_dataset(self.data_train, self.hparams.transform_train), 
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=True,
                            collate_fn=pad_list_data_collate)
        

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(dataset=self.get_transformed_dataset(self.data_val, self.hparams.transform_val), 
                            batch_size=self.hparams.batch_size, 
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=False,
                            collate_fn = pad_list_data_collate)
    
    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(dataset=self.get_transformed_dataset(self.data_val, self.hparams.transform_val), 
                            batch_size=self.hparams.batch_size, 
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=False,
                            collate_fn=pad_list_data_collate)
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def test_train_transform(self):
        return self.get_transformed_dataset(self.data_train, self.hparams.transform_train)

    def test_val_transform(self):
        return self.get_transformed_dataset(self.data_val, self.hparams.transform_val)

if __name__=="__main__":
    from omegaconf import DictConfig
    import hydra
    
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def main(cfg: DictConfig):
        datamodule = hydra.utils.instantiate(cfg.data)
        # print(cfg.data)
        datamodule.setup()
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))
        print(batch.keys())
        print(type(batch["image"]), type(batch["label"]))
        print(batch["label"].size())

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        datamodule = hydra.utils.instantiate(cfg.data)
        # print(cfg.data)
        datamodule.setup()
        transformed_data = datamodule.test_val_transform()
        print(datamodule.data_train[0])
        print(transformed_data[2]["image"].size())

    # main()
    test()