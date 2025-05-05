import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm
from collections.abc import Sequence

class PoolBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        super(PoolBlock, self).__init__()
        self.layers = nn.Sequential(
                                Convolution(spatial_dims=spatial_dims,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            dilation=strides,
                                            padding=None,
                                            adn_ordering="NDA",
                                            act="relu",
                                            norm=Norm.BATCH,
                                            dropout=dropout),
                                nn.MaxPool3d(strides, padding=0)
                                    )
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class ConvBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        super(ConvBlock, self).__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class AttentionEncoderBlock(nn.Module):
    def __init__(self,  spatial_dims: int,
                        in_channels: int,
                        out_channels: int,
                        kernel_size: list[int, int, int] | int = 3,
                        down_kernel_size: Sequence[int] | int = 3,
                        stride: list[int, int, int] | int = 1,
                        groups: int = 1,
                        dropout: float = 0.
                        ):
        super(AttentionEncoderBlock, self).__init__()
        padding = 0
        if isinstance(down_kernel_size, int):
            padding = down_kernel_size // 2
        else: 
            padding = [kernel // 2  for kernel in down_kernel_size]


        self.avg_pool = Convolution(spatial_dims=spatial_dims,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=down_kernel_size,
                                    strides=stride,
                                    padding=padding,
                                    dropout=dropout,
                                    norm=Norm.BATCH,
                                    act='relu'
                                    )
        self.max_pool = PoolBlock(spatial_dims=spatial_dims,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=down_kernel_size,
                                    strides=stride,
                                    dropout=dropout)

        self.attn = Convolution(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                dropout=dropout,
                                adn_ordering="NDA",
                                act='prelu',
                                norm=Norm.BATCH
                                )

    def forward(self,input: Tensor):
        x1 = self.avg_pool(input)
        x2 = self.max_pool(input)
        return self.attn(F.relu(x1 + x2))

# class AxialAttentionBlock(nn.Module):
#     def __init__(self, spatial_dims,
#                         in_channels,
#                         out_channels,
#                         kernel_size=3,
#                         stride=2,
#                         dropout=0.0):
#         super(AxialAttentionBlock, self).__init__()
    
#     def _create_block(self, spatial_dims, kernel_size, strides):
#         if isinstance(strides, int):
#             strides = [strides] * spatial_dims
#         for idx, stride in enumerate(strides):
#             kernel_size = 1
#         pass


class EncoderBlock(nn.Module):
    def __init__(self,  spatial_dims,
                        in_channels,
                        out_channels,
                        kernel_size: list[int] | int = 3,
                        down_kernel_size: Sequence[int] | int = 3,
                        stride: list[int] | int = 1,
                        groups: int = 4, 
                        dropout: float = 0.0
                    ):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(AttentionEncoderBlock(spatial_dims=spatial_dims, 
                                                        in_channels=in_channels, 
                                                        out_channels=out_channels, 
                                                        kernel_size=kernel_size,
                                                        down_kernel_size=down_kernel_size, 
                                                        stride=stride, 
                                                        groups=groups, 
                                                        dropout=dropout),
                                    Convolution(spatial_dims=spatial_dims,
                                                in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                strides=1,
                                                padding=None,
                                                adn_ordering="NDA",
                                                act="relu",
                                                norm=Norm.BATCH,
                                                dropout=dropout,
                                                ))
    
    def forward(self, x: Tensor):
        return self.layer(x)

class AttentionEncoder(nn.Module):
    def __init__(self,  spatial_dims: int,
                        in_channels: int,
                        channels: Sequence[int],
                        strides: Sequence[int] | int = 3,
                        kernel_size: Sequence[int] | int = 3,
                        down_kernel_size: Sequence[int] | int = 3,
                        groups: Sequence[int] | int = 4,
                        dropout: float = 0.0):
        super(AttentionEncoder, self).__init__()
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(channels)
        if isinstance(down_kernel_size, int):
            down_kernel_size = [down_kernel_size] * len(channels)
        if isinstance(groups, int):
            groups = [groups] * len(channels)
        
        self.layers = [ConvBlock(spatial_dims=spatial_dims,
                                in_channels=in_channels,
                                out_channels=channels[0],
                                kernel_size=kernel_size[0],
                                strides=1,
                                dropout=dropout)]
        for i in range(len(channels) - 1):
            self.layers.append(EncoderBlock(spatial_dims=spatial_dims,
                                                in_channels=channels[i],
                                                out_channels=channels[i+1],
                                                stride=strides[i],
                                                kernel_size=kernel_size[i+1],
                                                down_kernel_size=down_kernel_size[i],
                                                groups=groups[i],
                                                dropout=dropout))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, input: Tensor) -> Sequence[Tensor]:
        outputs = [input]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        
        return outputs[1:]

if __name__=="__main__":
    spatial_dims = 3
    in_channels = 3
    channels = [16, 32, 64, 128]
    strides = [[2, 2, 2], [1, 1, 2], [1, 2, 2]]
    kernel_size = 3
    down_kernel_size = 3
    groups = 4
    model = AttentionEncoder(spatial_dims=spatial_dims,
                                in_channels=in_channels,
                                channels=channels,
                                strides=strides,
                                kernel_size=kernel_size,
                                down_kernel_size=down_kernel_size,
                                groups=groups,
                                dropout=0.3)
    x = torch.randn(4, 3, 16, 256, 256, dtype=torch.float32)
    print(x.shape)
    model.to("cuda:0")
    x.to("cuda:0")
    f1, f2, f3, f4 = model(x)
    print(f1.shape, f2.shape, f3.shape, f4.shape)
