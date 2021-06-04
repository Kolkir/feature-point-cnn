from superpoint import SuperPoint
from synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import os


class BaseTrainer(object):
    def __init__(self, train_dataset, test_dataset, validation_dataset, batch_size):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size)

    def train_loop(self, model, loss_fn, optimizer):
        size = len(self.train_dataset)
        for batch, (image, true_points_map) in enumerate(self.train_dataloader):
            pointness_map, _ = model.forward(image)
            # image shape [batch_dim, channels = 1, h, w]
            loss = loss_fn(pointness_map, true_points_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(image)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, model, loss_fn):
        size = len(self.test_dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for image, true_points_map in self.test_dataloader:
                pointness_map, _ = model(image)
                test_loss += loss_fn(pointness_map, true_points_map).item()

        test_loss /= size
        print(f"Test Avg loss: {test_loss:>8f} \n")


class MagicPointTrainer(BaseTrainer):
    def __init__(self, synthetic_dataset_path, checkpoint_path, settings):
        self.settings = settings
        self.checkpoint_path = checkpoint_path
        self.train_dataset = SyntheticDataset(synthetic_dataset_path, settings, 'training')
        self.validation_dataset = SyntheticDataset(synthetic_dataset_path, settings, 'validation')
        self.test_dataset = SyntheticDataset(synthetic_dataset_path, settings, 'test')
        super(MagicPointTrainer, self).__init__(self.train_dataset, self.test_dataset, self.validation_dataset,
                                                self.settings.batch_size)
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs

    def train(self, model):
        optimizer = torch.optim.SGD(model.detector_parameters(), lr=self.learning_rate)
        loss = torch.nn.CrossEntropyLoss()

        def loss_fn(predicted, target):
            return loss(predicted, target)

        start_epoch = self.load_checkpoint(self.checkpoint_path, model, optimizer)

        for epoch in range(start_epoch, self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(model, loss_fn, optimizer)
            self.test_loop(model, loss_fn)
            self.save_checkpoint(epoch, model, optimizer, self.checkpoint_path)
        print("MagicPoint training done!")

    @staticmethod
    def load_checkpoint(path, model, optimizer):
        if os.path.exists(path):
            files = list(Path(path).glob('magic_point_*.pt'))
            if len(files) > 0:
                files.sort(reverse=True)
                filename = files[0]
                checkpoint = torch.load(filename)
                if checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    model.train()
                    return epoch
        return 0

    @staticmethod
    def save_checkpoint(epoch, model, optimizer, path):
        filename = os.path.join(path, 'magic_point_{0}.pt'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename)


class TrainWrapper(object):
    def __init__(self, checkpoint_path, synthetic_dataset_path, settings):
        self.checkpoint_path = checkpoint_path
        self.synthetic_dataset_path = synthetic_dataset_path
        self.settings = settings
        self.net = SuperPoint(self.settings)

    def train(self):
        self.net.disable_descriptor()
        magic_point_trainer = MagicPointTrainer(self.synthetic_dataset_path, self.checkpoint_path, self.settings)
        magic_point_trainer.train(self.net)
