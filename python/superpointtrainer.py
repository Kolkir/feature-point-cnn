import os

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from basetrainer import BaseTrainer
from netutils import get_points
from coco_dataset import CocoDataset
from saveutils import load_checkpoint, save_checkpoint


class SuperPointTrainer(BaseTrainer):
    def __init__(self, coco_dataset_path, checkpoint_path, settings):
        self.settings = settings
        self.checkpoint_path = checkpoint_path
        self.train_dataset = CocoDataset(coco_dataset_path, settings, 'train')
        self.test_dataset = CocoDataset(coco_dataset_path, settings, 'test')
        super(SuperPointTrainer, self).__init__(self.train_dataset, self.test_dataset, self.settings.batch_size)
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.train_iter = 0

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # TODO: Implement complex loss
        loss_fn = torch.nn.CrossEntropyLoss()

        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_checkpoint(self.checkpoint_path, model, optimizer)
        if prev_epoch > 0:
            start_epoch = prev_epoch + 1
        epochs_num = start_epoch + self.epochs

        self.train_iter = 0

        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(model, loss_fn, optimizer, epoch)
            self.test_loop(model, loss_fn, epoch)
            save_checkpoint(epoch, model, optimizer, self.checkpoint_path)

    def test_log_fn(self, loss_value, f1_value, image, pointness_map, n_iter):
        self.summary_writer.add_scalar('Loss/test', loss_value, n_iter)
        self.summary_writer.add_scalar('F1/test', f1_value, n_iter)
        # TODO: Implement
        # self.summary_writer.add_image('Detector result', res_img.transpose([2, 0, 1]), n_iter)

    def train_log_fn(self, loss, n_iter):
        self.summary_writer.add_scalar('Loss/train', loss, self.train_iter)
        self.train_iter += 1
