export WANDB_API_KEY=aa5821234bd41893b5da6713a5d8f3da4815ce68
export CUDA_VISIBLE_DEVICES=1
python '/work/hpc/spine-segmentation/src/train.py' logger=wandb trainer.max_epochs=300 \
                                                    trainer.check_val_every_n_epoch=2 \
                                                    # ckpt_path=/work/hpc/spine-segmentation/logs/train/runs/2024-08-08_15-34-59/checkpoints/epoch_265_v2.ckpt \
                                                    # logger.wandb.id=e7vgubjq