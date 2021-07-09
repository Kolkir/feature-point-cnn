import configparser
import unittest
import torchvision
import torch
from python.homographies import homography_adaptation, HomographyConfig
from python.netutils import get_points
from python.saveutils import load_checkpoint_for_inference
from python.settings import SuperPointSettings
from python.superpoint import SuperPoint


class TestHomographies(unittest.TestCase):

    def test_homography_adaptation(self):
        homo_config = HomographyConfig()
        config = configparser.ConfigParser()
        config.read('test.ini')
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

        img_h, img_w = image.shape[2], image.shape[3]

        probs, _, _ = net(image)
        points = get_points(probs, img_h, img_w, settings).T

        probs_with_adaptation = homography_adaptation(image, net, homo_config)
        points_with_adaptation = get_points(probs_with_adaptation, img_h, img_w, settings).T

        self.assertEqual(probs_with_adaptation.shape, torch.Size([1, img_h, img_w]),
                         'Output has an incorrect shape after the homography adaptation')

        self.assertEqual(probs_with_adaptation.shape, probs.shape,
                         'Probability maps have different sizes')

        self.assertAlmostEqual(points.shape[0], points_with_adaptation.shape[0], delta=50)

        self.assertLess(points.shape[0], 200, 'Too many points detections')


if __name__ == '__main__':
    unittest.main()
