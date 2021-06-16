import torch
import torchmetrics
from torch.utils.data import DataLoader


class BaseTrainer(object):
    def __init__(self, train_dataset, test_dataset, batch_size):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

    def train_loop(self, model, loss_fn, optimizer, n_iter):
        size = len(self.train_dataset)
        for batch, (image, true_points_map) in enumerate(self.train_dataloader):
            pointness_map, descriptors_map = model.forward(image)
            # image shape [batch_dim, channels = 1, h, w]
            if model.cuda():
                true_points_map = true_points_map.cuda()
            loss = loss_fn(pointness_map, descriptors_map, true_points_map)

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
        last_pointness_map = None
        with torch.no_grad():
            for image, true_points_map in self.test_dataloader:
                pointness_map, descriptors_map = model(image)
                if model.cuda():
                    true_points_map = true_points_map.cuda()
                test_loss += loss_fn(pointness_map, descriptors_map, true_points_map).item()

                softmax_result = softmax(pointness_map)
                f1 += f1_metric(softmax_result.cpu(), true_points_map.cpu())
                batches_num += 1

                last_pointness_map = pointness_map
                last_image = image

        test_loss /= batches_num
        f1 /= batches_num
        print(f"Test Avg loss: {test_loss:>8f} Avg F1:{f1:7f} \n")
        self.test_log_fn(test_loss, f1, last_image, last_pointness_map, n_iter)

    def test_log_fn(self, loss_value, f1_value, image, pointness_map, n_iter):
        pass

    def train_log_fn(self, loss_value, n_iter):
        pass
