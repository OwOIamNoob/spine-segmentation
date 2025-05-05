# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

import rootutils
rootutils.setup_root(search_from=__file__, indicator='setup.py', pythonpath=True)

from src.models.components.pos_embeddings import get_positional_embedding
from src.models.net.potato.encoder import AttentionEncoder
from src.models.net.potato.decoder import AttentionDecoder, UpConv

class PotatoNet(nn.Module):
    def __init__(self,
                    spatial_dims: int,
                    in_channels: int,
                    out_channels: int,
                    channels: Sequence[int],
                    strides: Sequence[int],
                    kernel_size: Sequence[int] | int = 3,
                    up_kernel_size: Sequence[int] | int = 3,
                    groups: Sequence[int] | int = 4,
                    dropout: float = 0.0
                ):
        super().__init__()
        self.emb = get_positional_embedding(spatial_dims, in_channels)
        # To cache positional encoding
        self.emb_encoding = None
        self.encoder = AttentionEncoder(spatial_dims=spatial_dims,
                                        in_channels=in_channels,
                                        channels=channels,
                                        strides=strides,
                                        kernel_size=kernel_size,
                                        down_kernel_size=up_kernel_size,
                                        groups=groups,
                                        dropout=dropout)
        self.decoder = AttentionDecoder(spatial_dims=spatial_dims,
                                        channels=channels,
                                        strides=strides,
                                        kernel_size=kernel_size,
                                        up_kernel_size=up_kernel_size,
                                        dropout=dropout)
        self.out_conv = Convolution(spatial_dims=spatial_dims,
                                    in_channels=channels[0],
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    strides=1,
                                    padding=0,
                                    conv_only=True)

    
    def forward(self, x:Tensor, supervision: bool = False, depth: int = 1) -> Sequence[Tensor] | Tensor:
        
        x += self.emb(x)
        fpns = self.encoder(x)
        features = self.decoder(fpns[:-1][::-1], fpns[-1], supervision=supervision, depth=depth)
        # print(len(features))
        output = self.out_conv(features[-1])
        if supervision is False:
            return output  
        else:  
            return output, features[:-1] 

class SupervisionModule(nn.Module):
    interpolation={ 1: "linear",
                    2: "bicubic",
                    3: "trilinear"}
    def __init__(self,  spatial_dims: int,
                        out_channels: int,
                        channels: Sequence[int]
                        ):
        super(SupervisionModule, self).__init__()
        self.layers = []
        for channel in channels:
            self.layers.append(Convolution(spatial_dims=spatial_dims,
                                            in_channels=channel,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding=0,
                                            conv_only=True))
        self.layers = nn.ModuleList(self.layers)
        self.interp = SupervisionModule.interpolation[spatial_dims]
    
    def forward(self, fpn: Sequence[Tensor]) -> Sequence[Tensor]:
        outputs = []
        for res, layer in zip(fpn, self.layers):
            outputs.append(layer(res))
        return outputs

class SupervisionPotatoNet(nn.Module):
    def __init__(self,
                    spatial_dims: int,
                    in_channels: int,
                    out_channels: int,
                    channels: Sequence[int],
                    strides: Sequence[int],
                    kernel_size: Sequence[int] | int = 3,
                    up_kernel_size: Sequence[int] | int = 3,
                    groups: Sequence[int] | int = 4,
                    dropout: float = 0.0,
                    depth: int = 3
                ):
        super(SupervisionPotatoNet, self).__init__()
        self.model = PotatoNet(spatial_dims=spatial_dims,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                channels=channels,
                                strides=strides,
                                kernel_size=kernel_size,
                                up_kernel_size=up_kernel_size,
                                groups=groups,
                                dropout=dropout)
        
        self.supervision = SupervisionModule(spatial_dims=spatial_dims,
                                                out_channels=out_channels,
                                                channels=channels[1:depth][::-1]
                                            )
        self.depth = depth
    
    def forward(self, input: Tensor, supervision: bool =False) -> Sequence[Tensor] | Tensor:
        if supervision is False:
            return self.model(input, supervision=False)

        output, features = self.model(input, supervision=True, depth=self.depth)
        sup_masks = self.supervision(features)
        return output, sup_masks[::-1]

        
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    @hydra.main(version_base="1.3", config_path="../../../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        model = hydra.utils.instantiate(cfg.model, _convert_='all')
        x = torch.randn(4, 1, 16, 256, 256, dtype=torch.float32)
        model.to("cuda:0")
        x = x.to("cuda:0")
        # supervision.to("cuda:0")
        output, sups = model.net(x, supervision=True)
        # masks = supervision(features)
        print(output.shape)
        print(len(sups))
        # # for feature in features:
        #     print(feature.shape)

        return True
    test()