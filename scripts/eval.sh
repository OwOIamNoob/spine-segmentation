export WANDB_API_KEY=610e7d7ec08264ac19d257213565223003941861
export CUDA_VISIBLE_DEVICES=$1
export HYDRA_FULL_ERROR=1
python '/work/hpc/spine-segmentation/src/eval.py'