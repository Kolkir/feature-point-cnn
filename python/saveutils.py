import os
from pathlib import Path
from python.weightsloader import load_weights_legacy
import torch


def load_checkpoint_for_inference(filename, model, load_legacy=False):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        if checkpoint:
            if load_legacy:
                miss_keys, _ = model.load_state_dict(load_weights_legacy(filename))
            else:
                miss_keys, _ = model.load_state_dict(checkpoint['model_state_dict'])
            if miss_keys:
                print('Can not load network some keys are missing:')
                print(miss_keys)
                exit(-1)

            model.eval()
            return True
    return False


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
