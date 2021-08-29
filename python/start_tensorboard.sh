#!/usr/bin/env bash

#LOG_DIR=../../magicpoint/runs
LOG_DIR=../../superpoint/runs
HOST=192.168.88.253

tensorboard --logdir $LOG_DIR --host $HOST
