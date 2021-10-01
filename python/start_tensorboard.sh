#!/usr/bin/env bash

if [ $1 = "magic" ]; then
LOG_DIR=../../magicpoint_data/checkpoints/runs
else
LOG_DIR=../../superpoint_data/checkpoints/runs
fi

HOST=192.168.88.253

echo "Start tracking: $LOG_DIR"
tensorboard --logdir $LOG_DIR --host $HOST
