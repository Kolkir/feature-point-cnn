import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, is_cuda, train_dataset, test_dataset, batch_size):
        self.is_cuda = is_cuda
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.use_amp = True
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train_loop(self, loss_fn, after_back_fn, optimizer):
        train_loss = 0
        for batch_index, batch in enumerate(tqdm(self.train_dataloader)):
            # See a loss function - now it uses different approach to clear gradients
            # optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(batch_index, *batch)

                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.scaler.scale(loss).backward()

                # Unscales gradients and calls
                # or skips optimizer.step()
                self.scaler.step(optimizer)

                # Updates the scale for next iteration
                self.scaler.update()
            else:
                loss = loss_fn(batch_index, *batch)
                loss.backward()
                optimizer.step()

            after_back_fn(loss, batch_index)
            train_loss += loss.item()

        train_loss /= len(self.train_dataloader)
        print(f"Train Avg loss: {train_loss:>8f} \n")

    def test_loop(self, loss_fn):
        test_loss = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(self.test_dataloader)):
                test_loss += loss_fn(*batch).item()

        test_loss /= len(self.test_dataloader)
        print(f"Test Avg loss: {test_loss:>8f} \n")
        return test_loss, len(self.test_dataloader)


