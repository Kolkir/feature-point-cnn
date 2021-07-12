import cv2
import numpy as np
import torch

from python.coco_dataset import CocoDataset
from python.netutils import get_points, restore_prob_map, make_prob_map_from_labels
from python.settings import SuperPointSettings


def draw_points(image, points, color):
    for point in points:
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 5, color, -1, lineType=16)


def test_coco_preporcess(coco_path):
    settings = SuperPointSettings()
    dataset = CocoDataset(coco_path, settings, 'training')
    for item in dataset:
        image, point_labels, warped_image, warped_point_labels, valid_mask, homography = item

        img_h, img_w = image.shape[1:]
        prob_map = make_prob_map_from_labels(point_labels, img_h, img_w, settings.cell)
        points = get_points(prob_map, img_h, img_w, settings)
        points = points.T

        # Draw result
        original_img = image.permute(1, 2, 0).data.cpu()
        original_img = cv2.UMat(original_img.numpy())
        draw_points(original_img, points, color=(0, 255, 0))

        cv2.imshow("Image", original_img)
        key = cv2.waitKey(delay=0)
        if key == ord('q'):
            break


if __name__ == '__main__':
    test_coco_preporcess('/mnt/a539f258-872a-4d38-966c-8cc04a2c3c9f/data_sets/coco')
