export WANDB_API_KEY=08610216d8143c15ddb5d2f16b6d432fefa2c827
export CUDA_VISIBLE_DEVICES=0
python '/work/hpc/spine-segmentation/src/train.py' logger=wandb trainer.max_epochs=300 \
                                                    trainer.check_val_every_n_epoch=2 \
                                                    logger.wandb.name=Refine_callback