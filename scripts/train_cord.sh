export WANDB_API_KEY=610e7d7ec08264ac19d257213565223003941861
export CUDA_VISIBLE_DEVICES=$1
export HYDRA_FULL_ERROR=1
python '/work/hpc/spine-segmentation/src/train_cord.py' model=spinal_cord logger=wandb trainer.max_epochs=300 \
                                                    trainer.check_val_every_n_epoch=2 \
                                                    logger.wandb.name=Cervical_Spinal_Cord_T1T2 \
                                                    # ckpt_path=logs/train/runs/2025-05-04_04-47-49/checkpoints/epoch_115.ckpt
                                                    # ckpt_path=logs/train/runs/2024-11-02_18-52-41/checkpoints/last.ckpt