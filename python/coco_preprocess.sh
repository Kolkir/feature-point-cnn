#!/usr/bin/env bash

COCO_PATH=/run/media/kirill/data/data_sets/coco
MAGIC_POINT_PT=/run/media/kirill/data/development/magicpoint_data/checkpoints/magic_point_33.pt

python main.py --cuda train --generate_points --coco_path $COCO_PATH --magic_point_weights $MAGIC_POINT_PT

