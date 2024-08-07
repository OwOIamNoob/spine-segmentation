from collections.abc import Callable, Hashable, Mapping
import monai
import SimpleITK as sitk
import numpy as np 
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform, MapTransform
# Spider has 16 classes total (maybe) instead of 4, so erm 
class ConvertToMultiChannelBasedOnSpiderClasses(Transform):
    """
    Convert labels to multi channels based on spider classes:
    0 -> 7 are the labels for each vetebrae in lower back 
    100 is the label for spinal canal
    201 -> 207 are the labels for each disk in lower back
    All segmentation mask is seperated
    """
    labels = [1, 2, 3, 4, 5, 6, 7, 100, 201, 202, 203, 204, 205, 206, 207]
    # labels = [1, 2, 4]
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        result = [img == label for label in ConvertToMultiChannelBasedOnSpiderClasses.labels]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)    


# Transformation wrapper
class ConvertToMultiChannelBasedOnSpiderClassesd(MapTransform):
    backend = ConvertToMultiChannelBasedOnSpiderClasses.backend
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnSpiderClasses()
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBasedOnSpiderClassesSemantic(Transform):
    """
    Convert labels to multi channels based on spider classes:
    0 -> 7 are the labels for each vetebrae in lower back 
    100 is the label for spinal canal
    201 -> 207 are the labels for each disk in lower back
    All segmentation mask is seperated
    """
    labels = [1, 2, 3, 4, 5, 6, 7, 100, 201, 202, 203, 204, 205, 206, 207]
    # labels = [1, 2, 4]
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)      
        # result = [img == label for label in ConvertToMultiChannelBasedOnSpiderClassesSemantic.labels]
        result = [(img // 100 == 0) & (img > 0), 
                  img // 100 == 1,
                  img // 100 == 2]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)    


# Transformation wrapper
class ConvertToMultiChannelBasedOnSpiderClassesdSemantic(MapTransform):
    backend = ConvertToMultiChannelBasedOnSpiderClassesSemantic.backend
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnSpiderClassesSemantic()
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d