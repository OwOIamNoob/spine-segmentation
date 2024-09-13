from collections.abc import Callable, Hashable, Mapping, Sequence
import monai
import SimpleITK as sitk
import numpy as np 

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform, MapTransform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype
from monai.data.meta_obj import get_track_meta

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

    def __init__(self):
        super().__init__()
    
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

# Interpolate background for transform invariance
class SemanticBackground(Transform):
    def __init__(self, include_background):
        super().__init__()
        self.active = include_background
    
    def __call__(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        if not self.active:
            return img 

        if isinstance(img, torch.Tensor):
            background,_ = torch.max(img, dim=0, keepdim=True)
            # print(background.shape, img.shape)
            img = torch.concatenate([1 - background, img], dim=0)
        else: 
            background = np.max(img, axis=0, keepdims=True)
            # print(background.shape, img.shape)
            img = np.concatenate([1 - background, img], dim=0)
        
        return img

class SemanticBackgroundd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, include_background=False):
        super().__init__(keys, allow_missing_keys)
        self.engine = SemanticBackground(include_background)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.engine(d[key])
        return d


class PercentileNormalizeIntensity(Transform):
    """
    Normalize input based on orthogonal but isotropic base of 
    Use calculated mean or std value of the input image if no `subtrahend` or `divisor` provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        percentile: Sequence | NdarrayOrTensor | None = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.percentile = percentile if percentile is not None else [0., 1.]
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    @staticmethod
    def _mean(x):
        if isinstance(x, np.ndarray):
            return np.mean(x)
        x = torch.mean(x.float())
        return x.item() if x.numel() == 1 else x

    def _std(self, x, mean):
        low, high = 1, 1
        (density, values) = torch.histogram(x, density=True)
        density = convert_to_tensor(density)
        density = torch.cumsum(density * (values[1:] - values[:-1]), dim=0)
        values = values[1:]
        low =  values[density > self.percentile[0]].min()
        high = values[density > self.percentile[1]].min()
        # print(low, mean, high)
        return mean - low, high - mean

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img, *_ = convert_data_type(img, dtype=torch.float32)

        if self.nonzero:
            slices = img != 0
            masked_img = img[slices]
            if not slices.any():
                return img
        else:
            slices = None
            masked_img = img

        _sub = self._mean(masked_img)
        _div_low, _div_high = self._std(masked_img, _sub)
        print(_div_low, _div_high)
        assert _div_low > 0, "Lower bound higher than mean"
        assert _div_high > 0, "Upper bound lower than mean"

        if isinstance(_sub, (torch.Tensor, np.ndarray)):
            _sub, *_ = convert_to_dst_type(_sub, img)
            if slices is not None:
                _sub = _sub[slices]
        # _div = div if div is not None else self._std(masked_img)
        # if np.isscalar(_div):
        #     if _div == 0.0:
        #         _div = 1.0
        # elif isinstance(_div, (torch.Tensor, np.ndarray)):
        #     _div_low, *_ = convert_to_dst_type(_div_low, img)
        #     _div_high, *_ = convert_to_dst_type(_div_high, img)
            # if slices is not None:
            #     _div = _div[slices]
            # _div[_div == 0.0] = 1.0

        _div = torch.ones_like(img)
        if slices is not None:
            _div = _div[slices]

        if slices is not None:
            img[slices] = (masked_img - _sub) 
            _div[img[slices] < 0] = _div_low
            _div[img[slices] > 0] = _div_high
            img[slices] /= _div
        else:
            img = (img - _sub) 
            _div[img < 0] = _div_low
            _div[img > 0] = _div_high
            img /= _div

        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._normalize(d)
        else:
            img = self._normalize(img)

        out = convert_to_dst_type(img, img, dtype=dtype)[0]
        return out

class PercentileNormalizeIntensityd(MapTransform):
    def __init__(self, keys: KeysCollection, 
                    allow_missing_keys: bool = False, 
                    percentile: Sequence | NdarrayOrTensor | None = None,
                    nonzero: bool = False,
                    channel_wise: bool = False,
                    dtype: DtypeLike = np.float32):
        super().__init__(keys, allow_missing_keys)
        self.engine = PercentileNormalizeIntensity(percentile=percentile,
                                                    nonzero=nonzero,
                                                    channel_wise=channel_wise,
                                                    dtype=dtype)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.engine(d[key])
        return d