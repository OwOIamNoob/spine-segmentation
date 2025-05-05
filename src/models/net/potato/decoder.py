from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

import rootutils
# rootutils.setup_root(search_from=__file__, indicator="setup.py", pythonpath=True)
from src.models.net.potato.encoder import ConvBlock


class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi + g * (1 - psi)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=out_channels, f_l=out_channels, f_int=out_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = ConvBlock(
            spatial_dims=spatial_dims, 
            in_channels=2 * out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size, 
            dropout=dropout
        )


    def forward(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(x)
        att = self.attention(g=fromlower, x=memory)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m
    
class AttentionDecoder(nn.Module):
    def __init__(self, spatial_dims: int,
                        channels: Sequence[int],
                        strides: Sequence[int] | int = 3,
                        kernel_size: Sequence[int] | int = 3,
                        up_kernel_size: Sequence[int] | int = 3,
                        dropout: float = 0.0):
        super().__init__()
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(channels)
        if isinstance(up_kernel_size, int):
            up_kernel_size = [up_kernel_size] * len(channels)

        self.layers = []
        for idx in list(range(1, len(channels)))[::-1]:
            self.layers.append(DecoderBlock(spatial_dims=spatial_dims,
                                            in_channels=channels[idx],
                                            out_channels=channels[idx-1],
                                            kernel_size=kernel_size[idx],
                                            up_kernel_size=up_kernel_size[idx-1],
                                            strides=strides[idx-1],
                                            dropout=dropout))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, mem: Sequence[Tensor], input: Tensor, supervision: bool = False, depth:int=1) -> Tensor:
        outputs = [input]
        for idx in range(len(mem)):
                if supervision:
                    outputs.append(self.layers[idx](mem[idx], outputs[-1]))
                else: 
                    outputs[0] = self.layers[idx](mem[idx], outputs[0])
        
        return outputs if supervision is False else outputs[-depth:] # Exclude since its the mainstream

if __name__ == "__main__":
    from src.models.net.potato.encoder import AttentionEncoder

    spatial_dims = 3
    in_channels = 3
    channels = [16, 32, 64, 128]
    strides = [[2, 2, 2], [1, 1, 2], [1, 2, 2]]
    kernel_size = 3
    down_kernel_size = 3
    groups = 4
    encoder = AttentionEncoder(spatial_dims=spatial_dims,
                                in_channels=in_channels,
                                channels=channels,
                                strides=strides,
                                kernel_size=kernel_size,
                                down_kernel_size=down_kernel_size,
                                groups=groups,
                                dropout=0.3)
    decoder = AttentionDecoder(spatial_dims=spatial_dims,
                                channels=channels,
                                strides=strides,
                                kernel_size=kernel_size,
                                up_kernel_size=down_kernel_size,
                                dropout=0.3,
                                supervision=True
                                )    
    x = torch.randn(4, 3, 16, 256, 256, dtype=torch.float32)
    print(x.shape)
    encoder.to("cuda:0")
    decoder.to("cuda:0")
    x.to("cuda:0")
    f1, f2, f3, f4 = encoder(x)
    print(f1.shape, f2.shape, f3.shape, f4.shape)
    d1, d2, d3 = decoder([f3, f2, f1], f4)
    print(d1.shape, d2.shape, d3.shape)

