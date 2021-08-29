#!/usr/bin/env bash

COCO_PATH=/data_sets/coco
CHECK_POINTS_PATH=/magicpoint

python main.py --cuda train --magic-point --coco-path $COCO_PATH  --checkpoint-path $CHECK_POINTS_PATH

