from collections.abc import Callable, Hashable, Mapping, Sequence
import monai
import SimpleITK as sitk
import numpy as np 

import torch
import torch.nn as nn

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

class ConvertToMultiChannelBasedOnSpiderSemanticClasses(Transform):
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

    def __init__(self, div, labels, dim=0):
        super().__init__()
        self.div = div
        self.labels = labels
        self.dim = dim
    
    def __call__(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        # print("input size: ", img.shape, img.dtype, img.unique())
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)      
        result = []
        # result = [img == label for label in ConvertToMultiChannelBasedOnSpiderSemanticClasses.labels]
        for label in self.labels: 
            if label == 0:
                result += [(img // self.div == label) & (img > 0)]
            else: 
                result += [img // self.div == label]
        # result = [(img // 100 == 0) & (img > 0), 
        #           img // 100 == 1,
        #           img // 100 == 2]
        return torch.stack(result, dim=self.dim).to(torch.float32) if isinstance(img, torch.Tensor) else np.stack(result, axis=self.dim).astype(np.float32)    

# Transformation wrapper
class ConvertToMultiChannelBasedOnSpiderSemanticClassesd(MapTransform):
    backend = ConvertToMultiChannelBasedOnSpiderSemanticClasses.backend
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, div=100, labels=[0, 1, 2], dim=0):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnSpiderSemanticClasses(div, labels, dim)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

# Interpolate background for transform invariance
class SemanticBackground(Transform):
    def __init__(self, include_background,
                recompute: bool = False,
                batch=False):
        super().__init__()
        self.active = include_background
        self.recompute = recompute
        self.channel = 0 if not batch else 1
    
    def __call__(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        # print(img.shape)
        if not self.active:
            return img 
        # Dropout background channel
        if self.recompute:
            if self.channel == 0:
                img = img[1:]
            else:
                img = img[:, 1:]
            
        if isinstance(img, torch.Tensor):
            background, _ = torch.max(img, dim=self.channel, keepdim=True)
            # print(background.shape, img.shape)
            img = torch.concatenate([1 - background, img], dim=self.channel)
        else: 
            background = np.max(img, axis=self.channel, keepdims=True)
            # print(background.shape, img.shape)
            img = np.concatenate([1 - background, img], dim=self.channel)
        
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

class SpiderInstanceToSemantic(Transform):
    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        bg = []
        if self.include_background is True:
            bg += [img[0]]
            img = img[1:] 
        
        if isinstance(img, np.ndarray):
            return np.stack(bg + [np.max(img[:7], axis=0), img[7], np.max(img[8:], axis=0)], axis=0)
        return torch.stack(bg + [torch.max(img[:7], dim=0)[0], img[7], torch.max(img[8:], dim=0)[0]], dim=0)

class SpiderInstanceToSemanticd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, include_background=False):
        super().__init__(keys, allow_missing_keys)
        self.engine = SpiderInstanceToSemantic(include_background)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.engine(d[key])
        return d

###########################         REMAP FUNCTION

class Remap(Transform):
    """ Remapping intensity range to predefined range
    """
    def __init__(
        self,
        quantiles: Sequence | NdarrayOrTensor | None = [0, 1],
        values: Sequence | NdarrayOrTensor | None = None,
        dst_values: Sequence | NdarrayOrTensor | None = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        quantile_downsampling: int = 2):
        super().__init__()
        self.quantiles = torch.Tensor(quantiles)
        self.nonzero = nonzero
        self.dtype = dtype
        self.dst_values = torch.Tensor(dst_values)
        self.values = values
        self.spacing = quantile_downsampling
        assert (values is None) ^ (quantiles is None), "Either value or quantile must be defined"
        
    
    def interp(self, img: NdarrayOrTensor, xp: NdarrayOrTensor, fp: NdarrayOrTensor) -> NdarrayOrTensor:
        # img = img.movedim(self.dim, -1)
        m = torch.diff(fp) / torch.diff(xp)
        b = fp[:-1] - m * xp[:-1] 
        # m = m[(None, )*(len(img.dim) - 1)]
        indices = torch.searchsorted(xp, img, right=False)
        indices = (indices - 1).clamp(0, m.shape[0] - 1)
        values = m[indices] * img + b[indices]
        return values 
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=self.dtype).contiguous()
        # print(type(img), type(self.quantiles))
        if self.nonzero: 
            mask = img == 0
        else: 
            mask = False
        # print(type(img), type(masked))
        if (~mask).sum() == 0:
            return img

        if not self.values:
            values = torch.quantile(img[~mask].flatten()[::self.spacing], self.quantiles)
        else: 
            values = self.values

        values = values.to(img.device)
        self.dst_values = self.dst_values.to(img.device)
        # print("Input range:", values)
        img = self.interp(img, values, self.dst_values)
        img[mask] = 0
        out = convert_to_dst_type(img, img, self.dtype)[0]
        # print("Output range", out.max(), out.min())
        return out

class Remapd(MapTransform):
    def __init__(self, keys: KeysCollection, 
                    allow_missing_keys: bool = False, 
                    quantiles: Sequence | NdarrayOrTensor | None = None,
                    values: Sequence | NdarrayOrTensor | None = None,
                    dst_values: Sequence | NdarrayOrTensor | None = None,
                    nonzero: bool = False,
                    dtype: DtypeLike = np.float32,
                    quantile_downsampling: int =2):

        super().__init__(keys, allow_missing_keys)
        self.engine = Remap(quantiles=quantiles, 
                            values=values,
                            dst_values=dst_values,
                            nonzero=nonzero,
                            dtype=dtype, 
                            quantile_downsampling=quantile_downsampling)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.engine(d[key])
        return d


####################            Intensity Scaling
class QuantileClipIntensity(Transform):
    """ Clip image intensity range based on its intensity distribution
    """
    def __init__(
        self,
        quantile: Sequence | NdarrayOrTensor | None = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32):
        super().__init__()
        self.quantile = torch.Tensor(quantile)
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
    
    def _clip(self, img:NdarrayOrTensor) -> NdarrayOrTensor:
        if self.nonzero:
            slices = img != 0
            if not slices.any():
                return img
            masked = img[slices]
        else: 
            slices = True
            masked = img
        low, high = torch.quantile(masked, self.quantile)
        img[(img < low) & slices] = low 
        img[(img > high) & slices] = high
        
        return img

    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._clip(d)
        else:
            img = self._clip(img)

        out = convert_to_dst_type(img, img, dtype=dtype)[0]
        return out

class QuantileClipIntensityd(MapTransform):
    def __init__(self, keys: KeysCollection, 
                    allow_missing_keys: bool = False, 
                    quantile: Sequence | NdarrayOrTensor | None = None,
                    nonzero: bool = False,
                    channel_wise: bool = False,
                    dtype: DtypeLike = np.float32):
        super().__init__(keys, allow_missing_keys)
        self.engine = QuantileClipIntensity(quantile=quantile,
                                            nonzero=nonzero,
                                            channel_wise=channel_wise,
                                            dtype=dtype)
    
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
        clip: bool = False,
        mean: float = 0.,
        std: float = 1.,
        dtype: DtypeLike = np.float32,
        quantile_downsampling: int = 2.
    ) -> None:
        self.percentile = percentile if percentile is not None else [0., 1.]
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.clip = clip
        self.mean = mean
        self.std = std
        self.spacing = quantile_downsampling

    @staticmethod
    def _mean(x):
        if isinstance(x, np.ndarray):
            return np.mean(x, dtype=float)
        x = torch.mean(x.float())
        return x.item() if x.numel() == 1 else x

    def _std(self, x, mean):
        # print(x.dtype)
        low, high = 1, 1
        (density, values) = torch.histogram(x, 200, density=True)
        density = convert_to_tensor(density)
        density = torch.cumsum(density * (values[1:] - values[:-1]), dim=0)

        values = values[1:]
        low =  values[density <= self.percentile[0]].max()
        high = values[density >= self.percentile[1]].min()
        # print(torch.mean(values * density), mean)
        assert low < mean, f"Lower bound higher than mean {low}, {mean}"
        assert high > mean, f"Upper bound lower than mean {high}, {mean}"
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

        quantiles = torch.quantile(masked_img.flatten()[::self.spacing], torch.Tensor(self.percentile))
        _sub = self._mean(masked_img)
        quantiles -= _sub
        _div_low, _div_high = -quantiles[0].item(), quantiles[1].item()
        # print(_sub, _div_low, _div_high)
        # print(_div_low, _div_high)

        if isinstance(_sub, (torch.Tensor, np.ndarray)):
            _sub, *_ = convert_to_dst_type(_sub, img)
            if slices is not None:
                _sub = _sub[slices]

        _div = torch.ones_like(img)
        if slices is not None:
            _div = _div[slices]

        if slices is not None:
            img[slices] = (masked_img - _sub) 
            _div[img[slices] < 0] = _div_low
            _div[img[slices] > 0] = _div_high
            img[slices] = (img[slices] / _div)
            if self.clip:
                w = img[slices].abs_()
                img[slices] = torch.where(w > 1., img[slices] / w, img[slices])
            img[slices] = img[slices] * self.std + self.mean
        else:
            img = (img - _sub) 
            _div[img < 0] = _div_low
            _div[img > 0] = _div_high
            img /= _div
            if self.clip:
                w = img.abs_()
                img = torch.where(w > 1., img / w, img)
            img = img * self.std + self.mean
            
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        
        img = convert_to_tensor(img, track_meta=get_track_meta())
        # print("Percentile Img shape", img.shape)
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
                    clip: bool = False,
                    mean: float = 0,
                    std: float = 1.,
                    dtype: DtypeLike = np.float32,
                    quantile_downsampling: int = 2):
        super().__init__(keys, allow_missing_keys)
        self.engine = PercentileNormalizeIntensity(percentile=percentile,
                                                    nonzero=nonzero,
                                                    channel_wise=channel_wise,
                                                    clip=clip,
                                                    mean=mean,
                                                    std=std,
                                                    dtype=dtype,
                                                    quantile_downsampling=quantile_downsampling)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.engine(d[key])
        return d

0
#### Post process

class InverseMask(Transform):
    def __init__(self, 
                channel: Sequence | int | None = None,
                batch: bool = True):
        super().__init__()
        self.channel = channel
        self.batch = batch
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.channel is None: 
            return 1 - channel
        if self.batch:
            img[:, self.channel] = 1 - img[:, self.channel] 
            # print(img[:, 0].to(dtype=float).mean())
        else: 
            img[self.channel] = 1 - img[self.channel]
            # print(img[0].to(dtype=float).mean())
        return img

class ImageOperation(Transform):
    def __init__(self,
                func: Callable | nn.Module | None = None,
                channel: Sequence | int | None = None,
                batch: bool = True):
        super().__init__()
        self.func = func
        if isinstance(channel, int):
            self.channel = [channel]
        else:
            self.channel = list(channel)
        self.batch = batch

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.func is None: 
            return img
        if self.channel is None:
            return self.func(img)
        
        if self.batch: 
            img[:, self.channel] = self.func(img[:, self.channel])
        else: 
            img[self.channel] = self.func(img[self.channel])
        
        return img

class BackgroundCrop(Transform):
    def __init__(self,
                channel: int = 0,
                batch: bool = True):
        super().__init__()
        self.channel = [channel]
        self.batch = batch
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.batch: 
            img *= img[:, self.channel]
        else: 
            img *= img[self.channel]
        
        return img
    
