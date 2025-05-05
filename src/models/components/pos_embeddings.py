from positional_encodings.torch_encodings import *
from collections.abc import Sequence
import torch 
from torch import Tensor
def get_positional_embedding(spatial_dims, channel, permute=True):
    match spatial_dims:
        case 1:
            return PositionalEncoding1D(channel) if not permute else PositionalEncodingPermute1D(channel)
        case 2:
            return PositionalEncoding2D(channel) if not permute else PositionalEncodingPermute2D(channel)
        case 3:
            return PositionalEncoding3D(channel) if not permute else PositionalEncodingPermute3D(channel)


class PotatoPositionalEmbedding(nn.Module):
    def __init__(self, spatial_dims: int, channel: int, group_dim: Sequence[int], magnitude: float = 1.):
        super(PotatoPositionalEmbedding, self).__init__()
        
        dim = spatial_dims - len(group_dim)
        roi = [ax for ax in list[range(spatial_dims)] if ax not in set(group_dim)]
        for d in group_dim:
            channel = channel * roi[d]
        self.group = group_dim
        pos_emb = get_positional_embedding(dim, channel)
        self.channel = channel
        self.ax = roi
        self.partial = None
        self.magnitude = magnitude

    @torch.no_grad()
    def forward(x: Tensor) -> Tensor:
        if isinstance(self.partial, Tensor):
            if self.partial.shape == x.shape: 
                return self.partial * self.magnitude
        size = torch.ones_like(x).reshape([x.shape[0], self.channel] + [x.shape[d + 2] for d in self.ax])
        self.partial = self.pos_emb(size).reshape(x.shape)
        return self.partial * self.magnitude