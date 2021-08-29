#!/usr/bin/env bash

SYNTHETIC_DATA_PATH=/data_sets/synthetic/
CHECK_POINTS_PATH=/magicpoint/

python main.py --cuda train --batch-size 32 --synthetic-path $SYNTHETIC_DATA_PATH  --checkpoint-path $CHECK_POINTS_PATH
