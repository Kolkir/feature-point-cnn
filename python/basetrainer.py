import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, is_cuda, train_dataset, test_dataset, batch_size, batch_size_divider):
        self.is_cuda = is_cuda
        self.batch_size_divider = batch_size_divider
        batch_size = batch_size // self.batch_size_divider
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        num_workers = 4
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers)
        self.use_amp = True
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train_loop(self, loss_fn, after_back_fn, optimizer):
        train_loss = torch.tensor(0., device='cuda' if self.is_cuda else 'cpu')
        batch_loss = torch.tensor(0., device='cuda' if self.is_cuda else 'cpu')
        prev_batch_index = 0
        real_batch_index = 0
        for batch_index, batch in enumerate(tqdm(self.train_dataloader)):
            # See a loss function - now it uses different approach to clear gradients
            # optimizer.zero_grad()
            optimizer_step_done = False
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(*batch)
                    # normalize loss to account for batch accumulation
                    loss = loss / self.batch_size_divider
                    train_loss += loss
                    batch_loss += loss

                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.scaler.scale(loss).backward()

                # gradient accumulation
                if ((batch_index + 1) % self.batch_size_divider == 0) or (
                        batch_index + 1 == len(self.train_dataloader)):
                    # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.step(optimizer)
                    # Updates the scale for next iteration
                    self.scaler.update()
                    optimizer_step_done = True
            else:
                loss = loss_fn(*batch)
                # normalize loss to account for batch accumulation
                loss /= self.batch_size_divider
                train_loss += loss
                loss.backward()
                # gradient accumulation
                if ((batch_index + 1) % self.batch_size_divider == 0) or (
                        batch_index + 1 == len(self.train_dataloader)):
                    optimizer.step()
                    optimizer_step_done = True

            # calculate statistics only after optimizer step to preserve graph
            if optimizer_step_done:
                if real_batch_index > prev_batch_index:
                    after_back_fn(batch_loss, real_batch_index)
                prev_batch_index = real_batch_index
                real_batch_index += 1
                batch_loss = 0.

        train_loss = train_loss.item() / real_batch_index
        print(f"Train Avg loss: {train_loss:>8f} \n")

    def test_loop(self, loss_fn):
        test_loss = 0
        batches_num = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(self.test_dataloader)):
                loss_value = loss_fn(*batch).item()

                # normalize loss to account for batch accumulation
                loss_value /= self.batch_size_divider
                test_loss += loss_value

                if ((batch_index + 1) % self.batch_size_divider == 0) or (
                        batch_index + 1 == len(self.train_dataloader)):
                    batches_num += 1

        test_loss /= batches_num
        print(f"Test Avg loss: {test_loss:>8f} \n")
        return test_loss, len(self.test_dataloader)
