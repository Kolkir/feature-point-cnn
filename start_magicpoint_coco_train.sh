#!/usr/bin/env bash

COCO_PATH=/root/coco
CHECK_POINTS_PATH=/root/magicpoint_checkpoints


python ./python/main.py --cuda train --magic_point --coco_path $COCO_PATH  --checkpoint_path $CHECK_POINTS_PATH

