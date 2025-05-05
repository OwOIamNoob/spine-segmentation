export WANDB_API_KEY=610e7d7ec08264ac19d257213565223003941861
export CUDA_VISIBLE_DEVICES=$1
export HYDRA_FULL_ERROR=1
python '/work/hpc/spine-segmentation/src/train.py' model=spider logger=wandb trainer.max_epochs=200 \
                                                    trainer.check_val_every_n_epoch=2 \
                                                    logger.wandb.name=SupervisionPotatoNet
                                                    # ckpt_path=logs/train/runs/2024-11-02_18-52-41/checkpoints/last.ckpt