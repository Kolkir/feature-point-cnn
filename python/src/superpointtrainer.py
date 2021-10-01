from src.basetrainer import BaseTrainer
from src.coco_dataset import CocoDataset
from src.losses import GlobalLoss
from src.saveutils import load_checkpoint_for_inference


class SuperPointTrainer(BaseTrainer):
    def __init__(self, coco_dataset_path, checkpoint_path, magic_point_weights, settings):
        self.magic_point_weights = magic_point_weights
        self.train_dataset = CocoDataset(coco_dataset_path, settings, 'train', do_augmentation=False)
        self.test_dataset = CocoDataset(coco_dataset_path, settings, 'test', size=1000, do_augmentation=False)
        super(SuperPointTrainer, self).__init__(settings, checkpoint_path, self.train_dataset, self.test_dataset)
        self.loss = GlobalLoss(self.settings)

    def train_init(self, check_point_loaded):
        if not check_point_loaded:
            # preload the MagicPoint state
            load_checkpoint_for_inference(self.magic_point_weights, self.model, ignore_missed=True)
            self.model.enable_descriptor()
            self.model.initialize_descriptor()

    def train_loss_fn(self, image, point_labels, warped_image, warped_point_labels, valid_mask,
                      homographies):
        prob_map, descriptors, point_logits = self.model.forward(image)
        warped_prob_map, warped_descriptors, warped_point_logits = self.model.forward(warped_image)

        # image shape [batch_dim, channels = 1, h, w]
        if self.settings.cuda:
            point_labels = point_labels.cuda()
            warped_point_labels = warped_point_labels.cuda()
            valid_mask = valid_mask.cuda()
            homographies = homographies.cuda()

        loss_value = self.loss(point_logits,
                               point_labels,
                               warped_point_logits,
                               warped_point_labels,
                               descriptors,
                               warped_descriptors,
                               homographies,
                               valid_mask)

        self.last_prob_map = prob_map
        self.last_labels = point_labels
        self.last_image = image
        self.last_warped_image = warped_image
        self.last_warped_prob_map = warped_prob_map
        self.last_warped_labels = warped_point_labels
        self.last_valid_mask = valid_mask

        return loss_value

    def test_loss_fn(self, image, point_labels, warped_image, warped_point_labels, valid_mask, homographies):
        _, descriptors, point_logits = self.model.forward(image)
        _, warped_descriptors, warped_point_logits = self.model.forward(warped_image)
        # image shape [batch_dim, channels = 1, h, w]
        if self.settings.cuda:
            point_labels = point_labels.cuda()
            warped_point_labels = warped_point_labels.cuda()
            valid_mask = valid_mask.cuda()
            homographies = homographies.cuda()

        loss_value = self.loss(point_logits,
                               point_labels,
                               warped_point_logits,
                               warped_point_labels,
                               descriptors,
                               warped_descriptors,
                               homographies,
                               valid_mask)

        return loss_value, point_logits
