from superpoint import SuperPoint
from synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
import torch


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
            for image, points in self.test_dataloader:
                points_prediction = model(image)
                test_loss += loss_fn(points_prediction, points).item()
                correct += (points_prediction.argmax(1) == points).to(torch.float32).sum().item()

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        

class MagicPointTrainer(BaseTrainer):
    def __init__(self, synthetic_dataset_path, settings):
        self.settings = settings
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

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_loop(model, loss_fn, optimizer)
            self.test_loop(model, loss_fn)
        print("MagicPoint training done!")


class TrainWrapper(object):
    def __init__(self, synthetic_dataset_path, settings):
        self.synthetic_dataset_path = synthetic_dataset_path
        self.settings = settings
        self.net = SuperPoint(self.settings)

    def train(self):
        self.net.disable_descriptor()
        magic_point_trainer = MagicPointTrainer(self.synthetic_dataset_path, self.settings)
        magic_point_trainer.train(self.net)
