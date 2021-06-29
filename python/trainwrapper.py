from magicpointtrainer import MagicPointTrainer
from superpointtrainer import SuperPointTrainer
from superpoint import SuperPoint


class TrainWrapper(object):
    def __init__(self, checkpoint_path, settings):
        self.checkpoint_path = checkpoint_path
        self.settings = settings
        self.net = SuperPoint(self.settings)
        self.net.train()
        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')

    def train_magic_point(self, synthetic_dataset_path):
        self.net.disable_descriptor()
        magic_point_trainer = MagicPointTrainer(synthetic_dataset_path, self.checkpoint_path, self.settings)
        magic_point_trainer.train(self.net)

    def train_super_point(self, coco_dataset_path):
        super_point_trainer = SuperPointTrainer(coco_dataset_path, self.checkpoint_path, self.settings)
        super_point_trainer.train(self.net)