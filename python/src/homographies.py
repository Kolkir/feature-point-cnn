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

import numpy as np
import torch
from scipy.stats import truncnorm
from torchvision.transforms import functional_tensor
import cv2


class HomographyConfig(object):
    def __init__(self):
        self.num = 10
        self.perspective = True
        self.scaling = True
        self.rotation = True
        self.translation = True
        self.n_scales = 5
        self.n_angles = 25
        self.scaling_amplitude = 0.1
        self.perspective_amplitude_x = 0.1
        self.perspective_amplitude_y = 0.1
        self.patch_ratio = 0.5
        self.max_angle = pi / 2
        self.allow_artifacts = False
        self.translation_overflow = 0.
        self.valid_border_margin = 8
        self.aggregation = 'sum'

    def init_for_preprocess(self):
        self.translation = True
        self.rotation = True
        self.scaling = True
        self.perspective = True
        self.scaling_amplitude = 0.2
        self.perspective_amplitude_x = 0.2
        self.perspective_amplitude_y = 0.2
        self.allow_artifacts = True
        self.patch_ratio = 0.85


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=torch.float32):
    a = mean - 2 * stddev
    b = mean + 2 * stddev
    return torch.tensor(truncnorm(a, b).rvs(shape), dtype=dtype)


def random_uniform(shape, low, high):
    if low > high:
        low, high = high, low
    if low == high:
        high = low + 0.00001
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
    x = torch.linalg.solve(a_mat, p_mat)
    homography = x.t()
    return homography.squeeze(dim=0)


def invert_homography(h):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(torch.linalg.inv(flat2mat(h)))


def flat2mat(h):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return torch.reshape(torch.cat([h, torch.ones([h.shape[0], 1], device=h.device)], dim=1), [-1, 3, 3])


def mat2flat(h):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    h = torch.reshape(h, [-1, 9])
    return (h / h[:, 8:9])[:, :8]


def homography_transform(t, h_coeffs, interpolation='bilinear'):
    return functional_tensor.perspective(t, h_coeffs.numpy().flatten(), interpolation=interpolation)


def homographic_augmentation(image, points, config):
    # Sample random homography transform
    img_h = image.shape[2]
    img_w = image.shape[3]
    image_shape = [img_h, img_w]
    homography = sample_homography(image_shape, config)
    #  Apply transformation
    warped_image = homography_transform(image, homography)
    valid_mask = compute_valid_mask(image_shape, homography,
                                    config.valid_border_margin)
    warped_points = warp_points(points, homography)
    warped_points = filter_points(warped_points, image_shape)

    return warped_image, warped_points, valid_mask, homography


def erode(image, erosion_radius):
    orig_device = image.device
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
    image = image.cpu().numpy().transpose([1, 2, 0])  # adapt channels for OpenCV format
    image = cv2.erode(image, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    image = np.expand_dims(image, axis=0)  # restore torch image format
    image = torch.from_numpy(image)
    image = image.to(device=orig_device)
    return image


def homography_adaptation(image, net, config):
    """ Performs homography adaptation.
    Inference using multiple random warped patches of the same input image for robust
    predictions.
    Arguments:
        image: A `Tensor` with shape `[B, C, H, W,]`.
        net: A function that takes an image as input, performs inference, and outputs the
            prediction dictionary.
        num: the number of sampled homographies.
        valid_border_margin: size of the border to ignore detections.
        aggregation: how to aggregate probabilities max or sum
    Returns:
        A dictionary which contains the aggregated detection probabilities.
    """

    all_probs, _, _ = net(image)
    all_counts = torch.ones_like(all_probs)

    all_probs.unsqueeze_(dim=-1)
    all_counts.unsqueeze_(dim=-1)

    shape = image.shape[2:4]

    def step(probs, counts):
        with torch.no_grad():
            # Sample image patch
            H = sample_homography(shape, perspective=config.perspective, scaling=config.scaling,
                                  rotation=config.rotation,
                                  translation=config.translation, n_scales=config.n_scales, n_angles=config.n_angles,
                                  scaling_amplitude=config.scaling_amplitude,
                                  perspective_amplitude_x=config.perspective_amplitude_x,
                                  perspective_amplitude_y=config.perspective_amplitude_y,
                                  patch_ratio=config.patch_ratio,
                                  max_angle=config.max_angle,
                                  allow_artifacts=config.allow_artifacts,
                                  translation_overflow=config.translation_overflow)
            H.unsqueeze_(dim=0)
            H_inv = invert_homography(H)
            warped = homography_transform(image, H)
            count = homography_transform(torch.ones(shape, device=image.device).unsqueeze(0),
                                         H_inv, interpolation='nearest')
            mask = homography_transform(torch.ones(shape, device=image.device).unsqueeze(0),
                                        H, interpolation='nearest')
            # Ignore the detections too close to the border to avoid artifacts
            if config.valid_border_margin != 0:
                count = erode(count, config.valid_border_margin)
                mask = erode(mask, config.valid_border_margin)

            # Predict detection probabilities
            warped_prob, _, _ = net(warped)
            warped_prob = warped_prob * mask
            warped_prob_proj = homography_transform(warped_prob, H_inv)
            warped_prob_proj = warped_prob_proj * count

            probs = torch.cat([probs, warped_prob_proj.unsqueeze(dim=-1)], dim=-1)
            count = count.repeat([image.shape[0], 1, 1])
            counts = torch.cat([counts, count.unsqueeze(dim=-1)], dim=-1)
            return probs, counts

    for i in range(config.num):
        all_probs, all_counts = step(all_probs, all_counts)

    all_counts = torch.sum(all_counts, dim=-1)
    max_prob = torch.max(all_probs, dim=-1)
    mean_prob = torch.sum(all_probs, dim=-1) / all_counts

    if config.aggregation == 'max':
        prob = max_prob
    elif config.aggregation == 'sum':
        prob = mean_prob
    else:
        raise ValueError(f'Unknown aggregation method: {config.aggregation}')

    prob = torch.where(all_counts >= config.num // 3, prob, torch.zeros_like(prob))

    return prob


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        image_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `int32` and shape (H, W).
    """
    mask = torch.ones(image_shape)
    if len(mask.shape) == 2:
        mask.unsqueeze_(dim=0)
    mask = homography_transform(mask, homography, interpolation='nearest')
    if erosion_radius > 0:
        mask = erode(mask, erosion_radius)
    return mask.to(dtype=torch.int32)


def warp_points(points, homography):
    """
    Warp a list of points with the INVERSE of the given homography.

    Arguments:
        points_homogenius: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 8) and (8,) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    h = homography.unsqueeze(dim=0) if len(homography.shape) == 1 else homography

    # Put the points into the homogeneous format
    num_points = points.shape[0]
    points_homogenius = points.to(dtype=torch.float32)
    points_homogenius[:, [0, 1]] = points_homogenius[:, [1, 0]]
    points_homogenius = torch.cat(
        [points_homogenius, torch.ones([num_points, 1], dtype=torch.float32, device=points_homogenius.device)], dim=-1)

    # # Apply the homography
    h_inv = flat2mat(invert_homography(h))
    warped_points = torch.tensordot(h_inv, points_homogenius, dims=[[2], [1]]).permute(0, 2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    warped_points[:, :, [0, 1]] = warped_points[:, :, [1, 0]]
    warped_points.squeeze_(0)

    return warped_points


def filter_points(points, shape):
    """
        Remove points laying out of the image shape
    """
    shape_tensor = torch.tensor(shape, dtype=torch.float) - 1
    mask = (points >= 0) & (points <= shape_tensor)
    mask = torch.all(mask, dim=1)
    filtered = points[mask]
    return filtered
