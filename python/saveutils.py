import os
from pathlib import Path

import torch


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
    return -1


def save_checkpoint(epoch, model, optimizer, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(path, 'magic_point_{0}.pt'.format(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
