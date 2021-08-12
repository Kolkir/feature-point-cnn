#!/usr/bin/env bash

COCO_PATH=/root/coco
MAGIC_POINT_PT=/root/magicpoint_checkpoints/magic_point_3.pt

python ./python/main.py --cuda train --generate_points --coco_path $COCO_PATH --magic_point_weights $MAGIC_POINT_PT 

