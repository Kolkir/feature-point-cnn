#!/usr/bin/env bash

WEIGHTS_PATH=/run/media/kirill/data/development/superpoint_data/checkpoints/super_point_7.pt

python main.py --cuda inference --weights_path $WEIGHTS_PATH