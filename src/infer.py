import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
# from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai import transforms
from monai.data.utils import list_data_collate
from src.data.components.spider_dataset import SpiderDataset, SpiderTransformedDataset
from torch.utils.data import DataLoader, Dataset
from src.models.brats21_module import Brats21LitModule

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="/data/hpc/spine/dataset/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test_brats21_1", type=str, help="experiment name")
parser.add_argument("--json_list", default="/data/hpc/spine/jsons/brats21_folds_test.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./runs/brats/",
    type=str,
    help="pretrained checkpoint directory",
)


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
        
    transform_val = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    dataset = SpiderDataset(data_dir=args.data_dir, json_path=args.json_list)

    test_loader = DataLoader(dataset=SpiderTransformedDataset(dataset, transform_val), 
                            batch_size=1, ##self.hparams.batch_size,
                            num_workers=4,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn = list_data_collate)
    
    
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = '/work/hpc/spine-segmentation/logs/train/runs/2024-04-04_15-08-49/checkpoints/last.ckpt' ##os.path.join(pretrained_dir, model_name)
    net = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    
    net.to(device)
    
    model = Brats21LitModule.load_from_checkpoint(pretrained_pth, net = net)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            print(batch.keys())
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            img_name = "BraTS2021_" + num + ".nii.gz"
            print("Inference on case {}".format(img_name))
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
        print("Finished inference!")


if __name__ == "__main__":
    main()