import torch
import torchmetrics
from torch.utils.data import DataLoader


class BaseTrainer(object):
    def __init__(self, is_cuda, train_dataset, test_dataset, batch_size):
        self.is_cuda = is_cuda
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

    def train_loop(self, model, loss_fn, optimizer, n_iter):
        size = len(self.train_dataset)
        for batch, (image, true_points, wraped_image, wraped_points, valid_mask) in enumerate(self.train_dataloader):
            _, descriptors, point_logits = model.forward(image)
            _, wraped_descriptors, wraped_point_logits = model.forward(wraped_image)
            # image shape [batch_dim, channels = 1, h, w]
            if self.is_cuda:
                true_points = true_points.cuda()
            loss = loss_fn(point_logits, true_points, descriptors, wraped_descriptors, valid_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(image)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                self.train_log_fn(loss, n_iter)
                
    def test_loop(self, model, loss_fn, n_iter):
        softmax = torch.nn.Softmax(dim=1)
        f1_metric = torchmetrics.F1(num_classes=65, mdmc_average='samplewise')

        test_loss = 0
        f1 = 0
        batches_num = 0

        last_image = None
        last_points = None
        with torch.no_grad():
            for image, true_points, wraped_image, wraped_points, valid_mask in self.test_dataloader:
                points_prob_map, descriptors, point_logits = model(image)
                _, wraped_descriptors, wraped_point_logits = model.forward(wraped_image)
                if self.is_cuda:
                    true_points = true_points.cuda()
                test_loss += loss_fn(point_logits, true_points, descriptors, wraped_descriptors, valid_mask).item()

                softmax_result = softmax(point_logits)
                f1 += f1_metric(softmax_result.cpu(), true_points.cpu())
                batches_num += 1

                last_points = points_prob_map
                last_image = image

        test_loss /= batches_num
        f1 /= batches_num
        print(f"Test Avg loss: {test_loss:>8f} Avg F1:{f1:7f} \n")
        self.test_log_fn(test_loss, f1, last_image, last_points, n_iter)

    def test_log_fn(self, loss_value, f1_value, image, points, n_iter):
        pass

    def train_log_fn(self, loss_value, n_iter):
        pass
