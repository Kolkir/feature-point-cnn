#!/usr/bin/env bash

SYNTHETIC_DATA_PATH=/run/media/kirill/data/data_sets/magicpoint/
CHECK_POINTS_PATH=/run/media/kirill/data/development/magicpoint_data/checkpoints

python main.py --cuda train --batch_size 32 --synthetic_path $SYNTHETIC_DATA_PATH  --checkpoint_path $CHECK_POINTS_PATH
