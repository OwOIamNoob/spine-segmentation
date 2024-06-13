#!/bin/bash
docker run -it \
--workdir="/work" \
--gpus all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--shm-size 4096 \
--volume="/data/hpc/spine/:/data" \
--volume="/work/hpc/spine-segmentation:/spine-segmentation" \
--volume="/work/hpc/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \
--name spine \
/bin/bash