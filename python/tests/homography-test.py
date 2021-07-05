import configparser
import unittest
import torchvision
import torch
from python.homographies import homography_adaptation, HomographyConfig
from python.saveutils import load_checkpoint_for_inference
from python.settings import SuperPointSettings
from python.superpoint import SuperPoint


class TestHomographies(unittest.TestCase):

    def test_homography_adaptation(self):
        homo_config = HomographyConfig()
        config = configparser.ConfigParser()
        config.read('homography-test.ini')
        settings = SuperPointSettings()
        settings.cuda = True
        net = SuperPoint(settings)
        load_checkpoint_for_inference(config['DEFAULT']['weights_path'], net)

        image = torchvision.io.read_image(config['DEFAULT']['image_path'], torchvision.io.image.ImageReadMode.GRAY)
        image = image.to(dtype=torch.float32) / 255.
        image.unsqueeze_(dim=0)  # add batch dimension

        if settings.cuda:
            net = net.cuda()
            image = image.cuda()
        outs = homography_adaptation(image, net, homo_config)
        img_h, img_w = image.shape[2], image.shape[3]
        self.assertEqual(outs.shape, torch.Size([1, img_h, img_w]),
                         'Output has an incorrect shape after the homography adaptation')


if __name__ == '__main__':
    unittest.main()
