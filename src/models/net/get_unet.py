import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import monai
from omegaconf import DictConfig, open_dict


def get_isotropic_attn_unet(config, func, net):
    roi = np.array(config.roi, dtype=int)
    print("Roi:", roi)
    spacing = np.array(config.spacing)
    ratio = spacing.max() / spacing
    log_ratio = np.ceil(np.log2(ratio)).astype(int)
    print("Ratio", log_ratio)
    iso_stride = np.ones([log_ratio.max(), 3], dtype=int)
    
    for i in range(3):
        iso_stride[:log_ratio[i], i] = 2
    print("Iso stride", iso_stride)
    roi = roi / np.power(2, log_ratio)
    print("Roi", roi)
    print("Bottleneck", config.bottleneck)
    zoom = np.floor(np.log2(roi / config.bottleneck)).astype(int).min()
    print("Zoom", zoom)
    stride = np.array([[2, 2, 2]] * zoom, dtype=int)
    stride = np.concatenate([iso_stride, stride])
    print("Total Stride", stride)
    # # channels = np.sqrt(stride[:, 0] * stride[:, 1] * stride[:, 2] / 2)
    # print(channels)
    channels = np.array([1] + [np.sqrt(np.prod(layer) / 2) for layer in stride])
    channels = np.cumprod(channels) * config.base_channel
    channels = np.floor(channels).astype(int)
    channels += channels % 2
    print(channels)
    # Update network config
    
    with open_dict(net):
        net._target_ = "monai.networks.nets.AttentionUnet"

    net.strides = stride.tolist()
    net.channels = channels.tolist()
    print(net)
    return func(net)


if __name__ == "__main__":
    import hydra
    import rootutils
    rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
    def test_config(cfg: DictConfig):
        # config = get_isotropic_attn_unet(cfg)
        # print(config)
        # print(type(config))
        # print(config["_target_"])
        # print(cfg.model.net)
        # net = get_isotropic_attn_unet(cfg.model.net.config, cfg.model.net.func, cfg.model.net.net)
        net = hydra.utils.instantiate(cfg.model.net)
        # print(net)

    test_config()