export WANDB_API_KEY=aa5821234bd41893b5da6713a5d8f3da4815ce68
export CUDA_VISIBLE_DEVICES=3
python '/work/hpc/spine-segmentation/src/train.py' experiment=spider callbacks=no_logger #logger=wandb