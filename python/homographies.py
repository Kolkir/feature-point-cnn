# The code is based on https://github.com/rpautrat/SuperPoint/ that is licensed as:
# MIT License
#
# Copyright (c) 2018 Paul-Edouard Sarlin & Rémi Pautrat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import pi
import torch
from scipy.stats import truncnorm
from torchvision.transforms import functional_tensor
import torchvision
import numpy as np
import sys
import cv2


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=torch.float32):
    a = mean - 2 * stddev
    b = mean + 2 * stddev
    return torch.tensor(truncnorm(a, b).rvs(shape), dtype=dtype)


def random_uniform(shape, low, high):
    return torch.distributions.uniform.Uniform(low, high).sample(shape)


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi / 2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.
    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    margin = (1 - patch_ratio) / 2
    pts1 = margin + torch.tensor([[0, 0],
                                  [0, patch_ratio],
                                  [patch_ratio, patch_ratio],
                                  [patch_ratio, 0]],
                                 dtype=torch.float32)
    # Corners of the input patch
    pts2 = pts1

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncated_normal([1], 0., perspective_amplitude_y / 2)
        h_displacement_left = truncated_normal([1], 0., perspective_amplitude_x / 2)
        h_displacement_right = truncated_normal([1], 0., perspective_amplitude_x / 2)
        pts2 += torch.stack([torch.cat([h_displacement_left, perspective_displacement], 0),
                             torch.cat([h_displacement_left, -perspective_displacement], 0),
                             torch.cat([h_displacement_right, perspective_displacement], 0),
                             torch.cat([h_displacement_right, -perspective_displacement], 0)])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = torch.cat(
            [torch.tensor([1.]), truncated_normal([n_scales], 1, scaling_amplitude / 2)], 0)
        center = torch.mean(pts2, dim=0, keepdim=True)
        scaled = torch.unsqueeze(pts2 - center, dim=0) * torch.unsqueeze(
            torch.unsqueeze(scales, dim=1), dim=1) + center
        if allow_artifacts:
            valid = torch.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = torch.nonzero(torch.sum((scaled >= 0.) & (scaled < 1.), [1, 2]))[:, 0]
        idx = valid[torch.randint(high=valid.shape[0], size=())]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, _ = torch.min(pts2, dim=0)
        t_max, _ = torch.min(1. - pts2, dim=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += torch.unsqueeze(torch.stack([random_uniform((), -t_min[0], t_max[0]),
                                             random_uniform((), -t_min[1], t_max[1])]),
                                dim=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = torch.linspace(-max_angle, max_angle, n_angles)
        angles = torch.cat([torch.tensor([0.]), angles], dim=0)  # in case no rotation is valid
        center = torch.mean(pts2, dim=0, keepdim=True)
        rot_mat = torch.reshape(torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles),
                                             torch.cos(angles)], dim=1), [-1, 2, 2])
        rotated = torch.matmul(
            torch.tile(torch.unsqueeze(pts2 - center, dim=0), [n_angles + 1, 1, 1]),
            rot_mat) + center
        if allow_artifacts:
            valid = torch.arange(n_angles)  # all angles are valid, except angle=0
        else:
            valid = torch.nonzero(torch.sum((rotated >= 0.) & (rotated < 1.), [1, 2]))[:, 0]
        idx = valid[torch.randint(high=valid.shape[0], size=())]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = torch.tensor(shape[::-1], dtype=torch.float32)  # different convention [y, x]
    pts1 *= torch.unsqueeze(shape, dim=0)
    pts2 *= torch.unsqueeze(shape, dim=0)

    def ax(p, q):
        return torch.tensor([p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]])

    def ay(p, q):
        return torch.tensor([0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]])

    a_mat = torch.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], dim=0)
    p_mat = torch.stack([pts2[i][j] for i in range(4) for j in range(2)]).t()
    p_mat.unsqueeze_(dim=1)
    x, _ = torch.solve(p_mat, a_mat)
    homography = x.t()
    return homography


def invert_homography(h):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(torch.inverse(flat2mat(h)))


def flat2mat(h):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return torch.reshape(torch.cat([h, torch.ones([h.shape[0], 1])], dim=1), [-1, 3, 3])


def mat2flat(h):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    h = torch.reshape(h, [-1, 9])
    return (h / h[:, 8:9])[:, :8]


def homography_transform(t, h_coeffs):
    return functional_tensor.perspective(t, h_coeffs.numpy().flatten())


def test_homography(image):
    # Sample random homography transform
    img_h = image.shape[1]
    img_w = image.shape[2]
    h = sample_homography([img_h, img_w])
    h_inv = invert_homography(h)

    # Apply transformation
    wrapped_image = homography_transform(image, h)
    restored_image = homography_transform(wrapped_image, h_inv)

    # Draw result
    original_img = image.permute(1, 2, 0).numpy()
    wrapped_img = wrapped_image.permute(1, 2, 0).numpy()
    restored_img = restored_image.permute(1, 2, 0).numpy()


    cv2.imshow("Original image", original_img)
    cv2.imshow("Wrapped image", wrapped_img)
    cv2.imshow("Restored image", restored_img)

    key = cv2.waitKey(delay=0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = torchvision.io.image.read_image(sys.argv[1])
        test_homography(img)
