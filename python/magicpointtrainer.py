import os

import cv2
import numpy as np
import torch
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from basetrainer import BaseTrainer
from netutils import get_points, make_prob_map_from_labels
from losses import DetectorLoss
from python.coco_dataset import CocoDataset
from synthetic_dataset import SyntheticDataset
from saveutils import save_checkpoint, load_last_checkpoint


class MagicPointTrainer(BaseTrainer):
    def __init__(self, dataset_path, checkpoint_path, settings, use_coco=False):
        self.settings = settings
        self.checkpoint_path = checkpoint_path
        if use_coco:
            self.train_dataset = CocoDataset(dataset_path, settings, 'train')
            self.test_dataset = CocoDataset(dataset_path, settings, 'test')
        else:
            self.train_dataset = SyntheticDataset(dataset_path, settings, 'training')
            self.test_dataset = SyntheticDataset(dataset_path, settings, 'test')
        super(MagicPointTrainer, self).__init__(self.settings.cuda, self.train_dataset, self.test_dataset,
                                                self.settings.batch_size)
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.train_iter = 0
        self.f1 = 0
        self.last_image = None
        self.last_prob_map = None
        self.last_labels = None

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss = DetectorLoss(self.settings.cuda)

        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_last_checkpoint(self.checkpoint_path, model, optimizer)
        if prev_epoch >= 0:
            start_epoch = prev_epoch + 1
        epochs_num = start_epoch + self.epochs

        self.train_iter = 0

        def train_loss_fn(image, true_points, *args):
            # This does not zero the memory of each individual parameter,
            # also the subsequent backward pass uses assignment instead of addition to store gradients,
            # this reduces the number of memory operations -compared to optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            prob_map, descriptors, point_logits = model.forward(image)
            # image shape [batch_dim, channels = 1, h, w]
            if self.is_cuda:
                true_points = true_points.cuda()
            loss_value = loss(point_logits, true_points, None)
            self.last_prob_map = prob_map
            self.last_labels = true_points
            self.last_image = image
            return loss_value

        def after_back_fn(loss_value, batch_index):
            if batch_index % 100 == 0:
                print(f"loss: {loss_value.item():>7f}")
                self.summary_writer.add_scalar('Loss/train', loss_value.item(), self.train_iter)

                if not model.grad_checkpointing:
                    for name, param in model.named_parameters():
                        if param.requires_grad and '_bn' not in name:
                            self.summary_writer.add_histogram(
                                tag=f"params/{name}", values=param, global_step=self.train_iter
                            )
                            self.summary_writer.add_histogram(
                                tag=f"grads/{name}", values=param.grad, global_step=self.train_iter
                            )

                img_h = self.last_image.shape[2]
                img_w = self.last_image.shape[3]
                points = get_points(self.last_prob_map[0, :, :].unsqueeze(dim=0).cpu(), img_h, img_w, self.settings)
                true_prob_map = make_prob_map_from_labels(self.last_labels[0, :, :].cpu().numpy(), img_h, img_w, self.settings.cell)
                true_points = get_points(true_prob_map[0, :, :].unsqueeze(dim=0), img_h, img_w, self.settings)
                frame = self.last_image[0, 0, :, :].cpu().numpy()
                res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
                for point in points.T:
                    point_int = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(res_img, point_int, 3, (255, 0, 0), -1, lineType=16)
                for point in true_points.T:
                    point_int = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)
                self.summary_writer.add_image('Detector result/train', res_img.transpose([2, 0, 1]), self.train_iter)

                self.train_iter += 1

        softmax = torch.nn.Softmax(dim=1)
        f1_metric = torchmetrics.F1(num_classes=65, mdmc_average='samplewise')

        def test_loss_fn(image, true_points, *args):
            points_prob_map, descriptors, point_logits = model(image)
            if self.is_cuda:
                true_points = true_points.cuda()
            loss_value = loss(point_logits, true_points, None)
            softmax_result = softmax(point_logits)
            self.f1 += f1_metric(softmax_result.cpu(), true_points.cpu())
            self.last_prob_map = points_prob_map
            self.last_image = image
            return loss_value

        scheduler = ExponentialLR(optimizer, gamma=0.9)
        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch}\n-------------------------------")
            self.train_loop(train_loss_fn, after_back_fn, optimizer)

            self.f1 = 0
            self.last_image = None
            self.last_prob_map = None
            test_loss, batches_num = self.test_loop(test_loss_fn)
            self.f1 /= batches_num
            print(f"Test Avg F1:{self.f1:7f} \n")
            self.test_log_fn(test_loss, epoch)

            save_checkpoint('magic_point', epoch, model, optimizer, self.checkpoint_path)
            scheduler.step()

    def test_log_fn(self, loss_value, n_iter):
        self.summary_writer.add_scalar('Loss/test', loss_value, n_iter)
        self.summary_writer.add_scalar('F1/test', self.f1, n_iter)
        img_h = self.last_image.shape[2]
        img_w = self.last_image.shape[3]
        points = get_points(self.last_prob_map[0, :, :].unsqueeze(dim=0), img_h, img_w, self.settings)
        frame = self.last_image[0, 0, :, :].cpu().numpy()
        res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
        for point in points.T:
            point_int = (int(round(point[0])), int(round(point[1])))
            cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)
        self.summary_writer.add_image('Detector result/test', res_img.transpose([2, 0, 1]), n_iter)
