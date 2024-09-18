export WANDB_API_KEY=610e7d7ec08264ac19d257213565223003941861
export CUDA_VISIBLE_DEVICES=$1
python '/work/hpc/spine-segmentation/src/train.py' logger=wandb trainer.max_epochs=300 \
                                                    trainer.check_val_every_n_epoch=2 \
                                                    logger.wandb.name=Entangled_Inversed_Dice_DTM
                                                    # ckpt_path=logs/train/runs/2024-09-16_12-47-18/checkpoints/last.ckpt