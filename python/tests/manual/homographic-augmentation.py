import sys
import numpy as np
import torch
import cv2
import torchvision

from python.homographies import HomographyConfig, homographic_augmentation, invert_homography, homography_transform


def draw_points(image, points, color):
    for point in points:
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 5, color, -1, lineType=16)


def test_homography(image):
    # Generate random feature points
    num_points = 20
    ys = torch.randint(0, image.shape[1], (num_points, ))
    xs = torch.randint(0, image.shape[2], (num_points, ))
    points = torch.stack([ys, xs], dim=1)

    # Sample random homography transform and apply transformation
    homography_config = HomographyConfig()
    warped_image, warped_points, valid_mask, homography = homographic_augmentation(image, points, homography_config)
    homography.unsqueeze_(dim=0)  # add batch size
    h_inv = invert_homography(homography)
    restored_image = homography_transform(warped_image, h_inv)

    # Draw result
    original_img = cv2.UMat(image.permute(1, 2, 0).numpy())
    warped_img = cv2.UMat(warped_image.permute(1, 2, 0).numpy())
    restored_img = restored_image.permute(1, 2, 0).numpy()
    mask_img = valid_mask.permute(1, 2, 0).numpy().astype(np.uint8)
    mask_img = mask_img * 255

    draw_points(original_img, points.numpy(), color=(0, 255, 0))
    draw_points(warped_img, warped_points.numpy(), color=(0, 0, 255))

    cv2.imshow("Original image", original_img)
    cv2.imshow("Warped image", warped_img)
    cv2.imshow("Restored image", restored_img)
    cv2.imshow("Mask", mask_img)

    key = cv2.waitKey(delay=0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img = torchvision.io.image.read_image(sys.argv[1])
        test_homography(img)