import sys

import torch
import cv2
import torchvision
import torchvision.transforms.functional as F

from src.homographies import HomographyConfig, homography_adaptation
from src.netutils import get_points
from src.saveutils import load_checkpoint_for_inference
from src.settings import SuperPointSettings
from src.superpoint import SuperPoint


def draw_points(image, points, color):
    for point in points:
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 2, color, -1, lineType=16)


def test_homographic_adaptation(weights_path, image_path):
    homo_config = HomographyConfig()
    settings = SuperPointSettings()
    settings.cuda = True
    net = SuperPoint(settings)
    net.eval()
    load_checkpoint_for_inference(weights_path, net)

    image = torchvision.io.read_image(image_path, torchvision.io.image.ImageReadMode.GRAY )
    image = image.to(dtype=torch.float32) / 255.

    # ratio preserving resize
    _, img_h, img_w = image.shape
    scale_h = 240 / img_h
    scale_w = 320 / img_w
    scale_max = max(scale_h, scale_w)
    new_size = [int(img_h * scale_max), int(img_w * scale_max)]
    image = F.resize(image, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    image = F.center_crop(image, [240, 320])

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
    original_img_with_adaptation = image.squeeze(dim=0).permute(1, 2, 0).data.cpu()
    original_img_with_adaptation = cv2.UMat(original_img_with_adaptation.numpy())

    draw_points(original_img, points, color=(255, 255, 255))
    draw_points(original_img_with_adaptation, points_with_adaptation, color=(255, 255, 255))

    cv2.imshow("Image", original_img)
    cv2.imshow("Adaptation", original_img_with_adaptation)
    key = cv2.waitKey(delay=0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
        image_path = sys.argv[2]
        test_homographic_adaptation(weights_path, image_path)
