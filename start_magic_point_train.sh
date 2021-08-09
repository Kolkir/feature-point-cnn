#!/usr/bin/env bash

SYNTHETIC_DATA_PATH=/run/media/kirill/data/data_sets/magicpoint/
CHECK_POINTS_PATH=/run/media/kirill/data/development/magicpoint_data/checkpoints

source ./python/venv/bin/activate

python ./python/main.py --cuda train --synthetic_path $SYNTHETIC_DATA_PATH  --checkpoint_path $CHECK_POINTS_PATH
