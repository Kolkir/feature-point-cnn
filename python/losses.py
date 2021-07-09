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

import torch
import numpy as np
from torch.nn.functional import log_softmax, normalize, relu
from python.homographies import warp_points


def masked_cross_entropy(logits, targets, class_weights, mask):
    log_probabilities = log_softmax(logits, dim=1)
    losses = -torch.gather(log_probabilities, dim=1, index=targets.unsqueeze(1))
    # weights
    weights = class_weights.view(-1, 1, 1)
    weights = weights.expand(-1, logits.size(2), logits.size(3))
    weights = torch.gather(weights, dim=0, index=targets)
    losses = losses * weights.unsqueeze(1)
    # apply mask
    batch_size = logits.shape[0]
    h = logits.shape[2]
    w = logits.shape[3]
    mask = torch.ones([batch_size, 1, h, w], dtype=torch.float32, device=logits.device) if mask is None else mask

    losses = losses * mask.float()
    losses_flat = losses.flatten(start_dim=1)
    loss = torch.sum(losses_flat, dim=1)
    loss = loss / torch.count_nonzero(mask.flatten(start_dim=1), dim=1)
    loss = torch.mean(loss)
    return loss


class DetectorLoss(object):
    def __init__(self, is_cuda):
        # configure class weights to better predict points of interest rather than empty ones,
        # because there are much more empty points
        self.class_weight = np.ones(65)  # 65 - when there no any point prediction for 8x8 patch
        self.class_weight[64] = 0.01
        self.class_weight = torch.from_numpy(self.class_weight).to(torch.float32)
        if is_cuda:
            self.class_weight = self.class_weight.cuda()

    def forward(self, points, true_points, valid_mask):
        if (valid_mask is not None) and len(valid_mask.shape) <= 2:  # skip empty masks
            valid_mask = None
        return masked_cross_entropy(points, true_points, self.class_weight, valid_mask)

    def __call__(self, points, true_points, valid_mask):
        return self.forward(points, true_points, valid_mask)


def descriptor_loss(descriptors, warped_descriptors, homographies, valid_mask, cell_size, positive_margin,
                    negative_margin, lambda_d):
    # Compute the position of the center pixel of every cell in the image
    batch_size = descriptors.shape[0]
    Hc = descriptors.shape[2]
    Wc = descriptors.shape[3]
    coord_cells = torch.stack(torch.meshgrid(torch.arange(0, Hc), torch.arange(0, Wc)), dim=-1)
    coord_cells = coord_cells.to(device=descriptors.device)
    coord_cells = coord_cells * cell_size + cell_size // 2  # (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points(coord_cells.view([-1, 2]), homographies)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = torch.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]).float()
    warped_coord_cells = torch.reshape(warped_coord_cells,
                                       [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1)
    s = torch.where(cell_distances < (cell_size - 0.5), 1., 0.)
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = torch.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
    descriptors = normalize(descriptors, dim=-1, p=2)
    warped_descriptors = torch.reshape(warped_descriptors,
                                       [batch_size, 1, 1, Hc, Wc, -1])
    warped_descriptors = normalize(warped_descriptors, dim=-1, p=2)
    dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=-1)
    dot_product_desc = relu(dot_product_desc)
    dot_product_desc = torch.reshape(normalize(
        torch.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
        dim=3, p=2), [batch_size, Hc, Wc, Hc, Wc])
    dot_product_desc = torch.reshape(normalize(
        torch.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
        dim=1, p=2), [batch_size, Hc, Wc, Hc, Wc])
    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    # Compute the loss
    zero = torch.tensor([0.], device=dot_product_desc.device)
    zero.expand_as(dot_product_desc)
    positive_dist = torch.maximum(zero, positive_margin - dot_product_desc)
    negative_dist = torch.maximum(zero, dot_product_desc - negative_margin)

    loss = lambda_d * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones([batch_size, 1, 1, Hc, Wc], dtype=torch.float32) if valid_mask is None else valid_mask
    valid_mask = torch.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = torch.sum(valid_mask) * float(Hc * Wc)
    loss = torch.sum(valid_mask * loss) / normalization
    return loss


class GlobalLoss(object):
    def __init__(self, is_cuda, lambda_loss, cell_size):
        self.detector_loss = DetectorLoss(is_cuda)
        self.lambda_loss = lambda_loss
        self.cell_size = cell_size

    def forward(self, points,
                true_points,
                warped_points,
                warped_true_points,
                descriptors,
                warped_descriptors,
                homographies,
                valid_mask):
        # Compute the loss for the detector head
        detector_loss_value = self.detector_loss.forward(points, true_points, valid_mask=None)
        warped_detector_loss_value = self.detector_loss.forward(
            warped_points, warped_true_points,
            valid_mask)

        # Compute the loss for the descriptor head
        descriptor_loss_value = descriptor_loss(
            descriptors, warped_descriptors, homographies, valid_mask, self.cell_size,
            lambda_d=250, positive_margin=1, negative_margin=0.2)

        loss = (detector_loss_value + warped_detector_loss_value
                + self.lambda_loss * descriptor_loss_value)
        return loss

    def __call__(self, points,
                 true_points,
                 warped_points,
                 warped_true_points,
                 descriptors,
                 warped_descriptors,
                 homographies,
                 valid_mask):
        return self.forward(points,
                            true_points,
                            warped_points,
                            warped_true_points,
                            descriptors,
                            warped_descriptors,
                            homographies,
                            valid_mask)
