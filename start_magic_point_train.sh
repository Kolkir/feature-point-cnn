#!/usr/bin/env bash

SYNTHETIC_DATA_PATH=/root/synthetic/
CHECK_POINTS_PATH=/root/magicpoint_checkpoints

python ./python/main.py --cuda train --batch_size 64 --synthetic_path $SYNTHETIC_DATA_PATH  --checkpoint_path $CHECK_POINTS_PATH
