import torch
from torch.utils.data import DataLoader


class BaseTrainer(object):
    def __init__(self, is_cuda, train_dataset, test_dataset, batch_size):
        self.is_cuda = is_cuda
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

    def train_loop(self, loss_fn, optimizer):
        train_loss = 0
        for batch_index, batch in enumerate(self.train_dataloader):
            loss = loss_fn(batch_index, *batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(self.train_dataloader)
        print(f"Test Avg loss: {train_loss:>8f} \n")

    def test_loop(self, loss_fn):
        test_loss = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(self.test_dataloader):
                test_loss += loss_fn(batch_index, *batch).item()

        test_loss /= len(self.test_dataloader)
        print(f"Test Avg loss: {test_loss:>8f} \n")
        return test_loss, len(self.test_dataloader)


