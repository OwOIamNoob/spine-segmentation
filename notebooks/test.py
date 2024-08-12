import torch
import rootutils
rootutils.setup_root(search_from="/work/hpc/spine-segmentation/notebooks/infer.ipynb", indicator="setup.py", pythonpath=True)

cp = torch.load("/work/hpc/spine-segmentation/logs/train/runs/2024-08-08_15-34-59/checkpoints/epoch_265.ckpt")
print(cp)
print(cp["state_dict"]["criterion.class_weight"])

cp["state_dict"]["criterion.class_weight"] = torch.Tensor([1])

torch.save(cp, "/work/hpc/spine-segmentation/logs/train/runs/2024-08-08_15-34-59/checkpoints/epoch_265_v2.ckpt")