#!/usr/bin/env bash

COCO_PATH=/data_sets/coco
MAGIC_POINT_PT=/magicpoint/magic_point_0.pt
CHECK_POINTS_PATH=/superpoint

python main.py --cuda train --batch-size-divider 16 --coco-path $COCO_PATH --magic-point-weights $MAGIC_POINT_PT --checkpoint-path $CHECK_POINTS_PATH

