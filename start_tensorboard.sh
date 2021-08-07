#!/usr/bin/env bash

LOG_DIR=../superpoint_data/checkpoints/runs
HOST=192.168.88.253

tensorboard --logdir $LOG_DIR --host $HOST
