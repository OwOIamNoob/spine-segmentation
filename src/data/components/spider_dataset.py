import monai
import SimpleITK as sitk
import numpy as np 
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import json
import os
# always starting with vanilla dataset, like its a norm to me now
class SpiderDataset(Dataset):
    num_class = 15
    def __init__(self, 
                 data_dir: str, 
                 json_path: str,
                 ):
        super().__init__()
        self.data = list()
        self.data_dir = data_dir
        self.setup(json_path)
    
    def setup(self, json_path):
        json_object = json.load(open(json_path, "r"))
        keys = json_object.keys()
        if "training" in keys:
            for key in keys:
                self.data.extend(json_object[key])
        else:
            try:
                self.data.extend(json_object)
            except:
                raise InsertionError("Something wrong with json file, cannot load or do anything, at all")
    
    def __getitem__(self, index):
        output = dict()
        output["input"] = ""
        output["output"] = ""
        if isinstance(self.data[index]["image"], list):
            output["input"] = [os.path.join(self.data_dir, image) for image in self.data[index]["image"]]
        else:
            output["input"] = os.path.join(self.data_dir, self.data[index]["image"])
        output["output"] = os.path.join(self.data_dir, self.data[index]["label"])
        return output
    def __len__(self) -> int:
        return len(self.data)
    
    

class SpiderTransformedDataset(Dataset):
    def __init__(self, 
                 dataset: SpiderDataset,
                 transform: monai.transforms.Compose):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        return self.transform(self.dataset[index])
    
    def __len__(self) -> int:
        return len(self.dataset)

if __name__=="__main__":
    dataset = SpiderDataset(data_dir = "./data/dataset/spine_nii", json_path="./data/jsons/spine_v2.json")
    transform = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["input", "output"]),
                                          monai.transforms.ConvertToMultiChannelBasedOnSpiderClassesd(keys=["output"]),
                                          monai.transforms.ToTensord(keys=["input", "output"]),])
    
    transformed = SpiderTransformedDataset(dataset, transform)
    data = dataset[0]
    images = transformed[0]
    print(data)
    print(images["input"].size(), images["input"].dtype)
    print(images["output"].size(), images["output"].dtype)
        
        