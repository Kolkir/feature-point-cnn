#!/usr/bin/env bash

COCO_PATH=/data_sets/coco
MAGIC_POINT_PT=/magicpoint/magic_point_0.pt

python main.py --cuda train --generate-points --coco_path $COCO_PATH --magic-point-weights $MAGIC_POINT_PT

