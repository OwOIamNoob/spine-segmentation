#!/bin/bash
image=$1
docker run -it \
--workdir="/work" \
--gpus all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--shm-size 4096 \
--volume="/data/hpc/spine/:/work/data" \
--volume="/work/hpc/spine-segmentation:/work/spine-segmentation" \
--volume="/work/hpc/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \
--name spine $image \
/bin/bash