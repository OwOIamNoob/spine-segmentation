from collections.abc import Callable, Hashable, Mapping
from typing import Dict, Union
import monai
import SimpleITK as sitk
import numpy as np 
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform, MapTransform



class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return img[None]

class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.adder = AddChannel()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d
