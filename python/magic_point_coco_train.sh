#!/usr/bin/env bash

COCO_PATH=/run/media/kirill/data/data_sets/coco
CHECK_POINTS_PATH=/run/media/kirill/data/development/magicpoint_data/checkpoints

python main.py --cuda train --magic_point --coco_path $COCO_PATH  --checkpoint_path $CHECK_POINTS_PATH

