from python.magicpointtrainer import MagicPointTrainer
from superpoint import SuperPoint


class TrainWrapper(object):
    def __init__(self, checkpoint_path, synthetic_dataset_path, settings):
        self.checkpoint_path = checkpoint_path
        self.synthetic_dataset_path = synthetic_dataset_path
        self.settings = settings
        self.net = SuperPoint(self.settings)
        self.net.train()
        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')

    def train(self):
        self.net.disable_descriptor()
        magic_point_trainer = MagicPointTrainer(self.synthetic_dataset_path, self.checkpoint_path, self.settings)
        magic_point_trainer.train(self.net)
