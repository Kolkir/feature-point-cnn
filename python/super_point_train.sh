#!/usr/bin/env bash

COCO_PATH=/run/media/kirill/data/data_sets/coco
MAGIC_POINT_PT=/run/media/kirill/data/development/magicpoint_data/checkpoints/magic_point_7.pt
CHECK_POINTS_PATH=/run/media/kirill/data/development/superpoint_data/checkpoints

python main.py --cuda train --batch_size_divider 16 --coco_path $COCO_PATH --magic_point_weights $MAGIC_POINT_PT --checkpoint_path $CHECK_POINTS_PATH

