import os

import torch
from torch.utils.tensorboard import SummaryWriter

from basetrainer import BaseTrainer
from coco_dataset import CocoDataset
from python.losses import GlobalLoss
from saveutils import save_checkpoint, load_last_checkpoint, load_checkpoint_for_inference


class SuperPointTrainer(BaseTrainer):
    def __init__(self, coco_dataset_path, checkpoint_path, magic_point_weights, settings):
        self.settings = settings
        self.magic_point_weights = magic_point_weights
        self.checkpoint_path = checkpoint_path
        self.train_dataset = CocoDataset(coco_dataset_path, settings, 'train')
        self.test_dataset = CocoDataset(coco_dataset_path, settings, 'train')  # TODO: change to 'test'
        super(SuperPointTrainer, self).__init__(self.settings.cuda, self.train_dataset, self.test_dataset,
                                                self.settings.batch_size)
        print(f'Trainer is initialized with batch size = {self.settings.batch_size}')
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.train_iter = 0

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss = GlobalLoss(self.settings.cuda, lambda_loss=0.0001, settings=self.settings)

        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_last_checkpoint(self.checkpoint_path, model, optimizer)
        if prev_epoch > 0:
            start_epoch = prev_epoch + 1
        else:
            # preload the MagicPoint state
            load_checkpoint_for_inference(self.magic_point_weights, model)
            model.initialize_descriptor()
        epochs_num = start_epoch + self.epochs

        self.train_iter = 0

        def train_loss_fn(batch_index, image, point_labels, warped_image, warped_point_labels, valid_mask,
                          homographies):
            # This does not zero the memory of each individual parameter,
            # also the subsequent backward pass uses assignment instead of addition to store gradients,
            # this reduces the number of memory operations -compared to optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
                
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

            return loss_value

        def after_back_fn(loss_value, batch_index):
            if batch_index % 100 == 0:
                print(f"loss: {loss_value.item():>7f}")
                self.summary_writer.add_scalar('Loss/train', loss_value.item(), self.train_iter)
                self.train_iter += 1

                if not model.grad_checkpointing:
                    for name, param in model.named_parameters():
                        if param.requires_grad and '_bn' not in name:
                            self.summary_writer.add_histogram(
                                tag=f"params/{name}", values=param, global_step=self.train_iter
                            )
                            self.summary_writer.add_histogram(
                                tag=f"grads/{name}", values=param.grad, global_step=self.train_iter
                            )

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
            return loss_value

        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(train_loss_fn, after_back_fn, optimizer)
            # TODO: restore
            # self.test_loop(test_loss_fn)
            save_checkpoint('super_point', epoch, model, optimizer, self.checkpoint_path)
