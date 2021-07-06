import os

import cv2
import numpy as np
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from basetrainer import BaseTrainer
from netutils import get_points
from losses import DetectorLoss
from synthetic_dataset import SyntheticDataset
from saveutils import load_checkpoint, save_checkpoint


class MagicPointTrainer(BaseTrainer):
    def __init__(self, synthetic_dataset_path, checkpoint_path, settings):
        self.settings = settings
        self.checkpoint_path = checkpoint_path
        self.train_dataset = SyntheticDataset(synthetic_dataset_path, settings, 'training')
        self.test_dataset = SyntheticDataset(synthetic_dataset_path, settings, 'test')
        super(MagicPointTrainer, self).__init__(self.settings.cuda, self.train_dataset, self.test_dataset,
                                                self.settings.batch_size)
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.train_iter = 0
        self.f1 = 0
        self.last_image = None
        self.last_prob_map = None

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss = DetectorLoss(self.settings.cuda)

        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_checkpoint(self.checkpoint_path, model, optimizer)
        if prev_epoch > 0:
            start_epoch = prev_epoch + 1
        epochs_num = start_epoch + self.epochs

        self.train_iter = 0

        def train_loss_fn(batch_index, image, true_points):
            _, descriptors, point_logits = model.forward(image)
            # image shape [batch_dim, channels = 1, h, w]
            if self.is_cuda:
                true_points = true_points.cuda()
            loss_value = loss(point_logits, true_points, None)

            if batch_index % 100 == 0:
                print(f"loss: {loss_value.item():>7f}")
                self.summary_writer.add_scalar('Loss/train', loss_value.item(), self.train_iter)
                self.train_iter += 1

            return loss_value

        softmax = torch.nn.Softmax(dim=1)
        f1_metric = torchmetrics.F1(num_classes=65, mdmc_average='samplewise')

        def test_loss_fn(image, true_points):
            points_prob_map, descriptors, point_logits = model(image)
            if self.is_cuda:
                true_points = true_points.cuda()
            loss_value = loss(point_logits, true_points, None)
            softmax_result = softmax(point_logits)
            self.f1 += f1_metric(softmax_result.cpu(), true_points.cpu())
            self.last_prob_map = points_prob_map
            self.last_image = image
            return loss_value

        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(train_loss_fn, optimizer)

            self.f1 = 0
            self.last_image = None
            self.last_prob_map = None
            test_loss, batches_num = self.test_loop(test_loss_fn)
            self.f1 /= batches_num
            print(f"Test Avg F1:{self.f1:7f} \n")
            self.test_log_fn(test_loss,  epoch)

            save_checkpoint('magic_point', epoch, model, optimizer, self.checkpoint_path)

    def test_log_fn(self, loss_value, n_iter):
        self.summary_writer.add_scalar('Loss/test', loss_value, n_iter)
        self.summary_writer.add_scalar('F1/test', self.f1, n_iter)
        img_h = self.last_image.shape[2]
        img_w = self.last_image.shape[3]
        points = get_points(self.last_prob_map[0, :, :], img_h, img_w, self.settings)
        frame = self.last_image[0, 0, :, :].cpu().numpy()
        res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
        for point in points.T:
            point_int = (int(round(point[0])), int(round(point[1])))
            cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)
        self.summary_writer.add_image('Detector result', res_img.transpose([2, 0, 1]), n_iter)
