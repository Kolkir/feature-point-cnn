import os

import cv2
import numpy as np
import torch
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from basetrainer import BaseTrainer
from coco_dataset import CocoDataset
from python.losses import GlobalLoss
from python.netutils import get_points, make_prob_map_from_labels, make_points_labels
from saveutils import save_checkpoint, load_last_checkpoint, load_checkpoint_for_inference


class SuperPointTrainer(BaseTrainer):
    def __init__(self, coco_dataset_path, checkpoint_path, magic_point_weights, settings):
        self.settings = settings
        self.magic_point_weights = magic_point_weights
        self.checkpoint_path = checkpoint_path
        self.train_dataset = CocoDataset(coco_dataset_path, settings, 'train')
        self.test_dataset = CocoDataset(coco_dataset_path, settings, 'test', size=1000)
        super(SuperPointTrainer, self).__init__(self.settings.cuda, self.train_dataset, self.test_dataset,
                                                self.settings.batch_size, self.settings.batch_size_divider)
        print(f'Trainer is initialized with batch size = {self.settings.batch_size}')
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.f1 = 0
        self.train_iter = 0
        self.last_image = None
        self.last_prob_map = None
        self.last_labels = None
        self.last_warped_image = None
        self.last_warped_prob_map = None
        self.last_warped_labels = None
        self.last_valid_mask = None

    def add_mask_image_summary(self, name, mask, labels, prob_map):
        img_h = prob_map.shape[1]
        img_w = prob_map.shape[2]
        points = get_points(prob_map[0, :, :].unsqueeze(dim=0).cpu(), img_h, img_w, self.settings)
        points = points.T
        points[:, [0, 1]] = points[:, [1, 0]]
        predictions = make_points_labels(points, img_h, img_w, self.settings.cell)

        frame_predictions = (predictions != 64)
        frame_labels = (labels[0, :, :] != 64).cpu().numpy()
        frame = mask[0, 0, :, :].cpu().numpy()
        res_img = (np.dstack((frame, frame_labels, frame_predictions)) * 255.).astype('uint8')
        self.summary_writer.add_image(f'Detector {name} result/train', res_img.transpose([2, 0, 1]), self.train_iter)

    def add_image_summary(self, name, image, prob_map, labels):
        img_h = image.shape[2]
        img_w = image.shape[3]
        points = get_points(prob_map[0, :, :].unsqueeze(dim=0).cpu(), img_h, img_w, self.settings)
        true_prob_map = make_prob_map_from_labels(labels[0, :, :].cpu().numpy(), img_h, img_w,
                                                  self.settings.cell)
        true_points = get_points(true_prob_map[0, :, :].unsqueeze(dim=0), img_h, img_w, self.settings)
        frame = image[0, 0, :, :].cpu().numpy()
        res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
        for point in points.T:
            point_int = (int(round(point[0])), int(round(point[1])))
            cv2.circle(res_img, point_int, 3, (255, 0, 0), -1, lineType=16)
        for point in true_points.T:
            point_int = (int(round(point[0])), int(round(point[1])))
            cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)
        self.summary_writer.add_image(f'Detector {name} result/train', res_img.transpose([2, 0, 1]), self.train_iter)

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss = GlobalLoss(self.settings.cuda, lambda_loss=0.1, settings=self.settings)

        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_last_checkpoint(self.checkpoint_path, model, optimizer)
        if prev_epoch >= 0:
            start_epoch = prev_epoch + 1
        else:
            # preload the MagicPoint state
            load_checkpoint_for_inference(self.magic_point_weights, model)
            model.enable_descriptor()
            model.initialize_descriptor()
        epochs_num = start_epoch + self.epochs

        self.train_iter = 0

        def train_loss_fn(image, point_labels, warped_image, warped_point_labels, valid_mask,
                          homographies):
            # This does not zero the memory of each individual parameter,
            # also the subsequent backward pass uses assignment instead of addition to store gradients,
            # this reduces the number of memory operations -compared to optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            prob_map, descriptors, point_logits = model.forward(image)
            warped_prob_map, warped_descriptors, warped_point_logits = model.forward(warped_image)

            # image shape [batch_dim, channels = 1, h, w]
            if self.is_cuda:
                point_labels = point_labels.cuda()
                warped_point_labels = warped_point_labels.cuda()
                valid_mask = valid_mask.cuda()
                homographies = homographies.cuda()

            loss_value = loss(point_logits,
                              point_labels,
                              warped_point_logits,
                              warped_point_labels,
                              descriptors,
                              warped_descriptors,
                              homographies,
                              valid_mask)

            self.last_prob_map = prob_map
            self.last_labels = point_labels
            self.last_image = image
            self.last_warped_image = warped_image
            self.last_warped_prob_map = warped_prob_map
            self.last_warped_labels = warped_point_labels
            self.last_valid_mask = valid_mask

            return loss_value

        def after_back_fn(loss_value, batch_index):
            if batch_index % 100 == 0:
                print(
                    f"loss: {loss_value.item():>7f}, detector {loss.detector_loss_value.item()}, warped detector {loss.warped_detector_loss_value.item()}, descriptor {loss.descriptor_loss_value.item()}")
                self.summary_writer.add_scalar('Loss/train', loss_value.item(), self.train_iter)

                for name, param in model.named_parameters():
                    if param.grad is not None and '_bn' not in name:
                        self.summary_writer.add_histogram(
                            tag=f"params/{name}", values=param, global_step=self.train_iter
                        )
                        self.summary_writer.add_histogram(
                            tag=f"grads/{name}", values=param.grad, global_step=self.train_iter
                        )

                self.add_image_summary('normal', self.last_image, self.last_prob_map, self.last_labels)
                self.add_image_summary('warped', self.last_warped_image, self.last_warped_prob_map,
                                       self.last_warped_labels)
                self.add_mask_image_summary('mask', self.last_valid_mask, self.last_warped_labels,
                                            self.last_warped_prob_map)

                self.train_iter += 1

        softmax = torch.nn.Softmax(dim=1)
        f1_metric = torchmetrics.F1(num_classes=65, mdmc_average='samplewise')

        def test_loss_fn(image, point_labels, warped_image, warped_point_labels, valid_mask, homographies):
            _, descriptors, point_logits = model.forward(image)
            _, warped_descriptors, warped_point_logits = model.forward(warped_image)
            # image shape [batch_dim, channels = 1, h, w]
            if self.is_cuda:
                point_labels = point_labels.cuda()
                warped_point_labels = warped_point_labels.cuda()
                valid_mask = valid_mask.cuda()
                homographies = homographies.cuda()

            loss_value = loss(point_logits,
                              point_labels,
                              warped_point_logits,
                              warped_point_labels,
                              descriptors,
                              warped_descriptors,
                              homographies,
                              valid_mask)

            softmax_result = softmax(point_logits)
            self.f1 += f1_metric(softmax_result.cpu(), point_labels.cpu())

            return loss_value

        scheduler = ExponentialLR(optimizer, gamma=0.9)
        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(train_loss_fn, after_back_fn, optimizer)

            self.f1 = 0
            self.last_image = None
            self.last_prob_map = None
            test_loss, batches_num = self.test_loop(test_loss_fn)
            self.f1 /= batches_num
            print(f"Test Avg F1:{self.f1:7f} \n")
            self.summary_writer.add_scalar('Loss/test', test_loss, epoch)
            self.summary_writer.add_scalar('F1/test', self.f1, epoch)

            save_checkpoint('super_point', epoch, model, optimizer, self.checkpoint_path)
            scheduler.step()
