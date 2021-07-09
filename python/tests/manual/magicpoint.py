import configparser
import torch
import cv2
import torchvision

from python.homographies import HomographyConfig, homography_adaptation
from python.netutils import get_points
from python.saveutils import load_checkpoint_for_inference
from python.settings import SuperPointSettings
from python.superpoint import SuperPoint


def draw_points(image, points, color):
    for point in points:
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 5, color, -1, lineType=16)


def test_magic_point():
    homo_config = HomographyConfig()
    config = configparser.ConfigParser()
    config.read('../test.ini')
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

    probs_with_adaptation = homography_adaptation(image, net, homo_config)
    points_with_adaptation = get_points(probs_with_adaptation, img_h, img_w, settings).T

    probs, _, _ = net(image)
    points = get_points(probs, img_h, img_w, settings).T

    # Draw result
    original_img = image.squeeze(dim=0).permute(1, 2, 0).data.cpu()
    original_img = cv2.UMat(original_img.numpy())
    draw_points(original_img, points, color=(0, 255, 0))
    draw_points(original_img, points_with_adaptation, color=(255, 255, 255))

    cv2.imshow("Image", original_img)
    key = cv2.waitKey(delay=0)


if __name__ == '__main__':
    test_magic_point()
