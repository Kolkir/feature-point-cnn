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
from torch.nn.functional import log_softmax, normalize, relu
from src.homographies import warp_points
import torch.nn.functional as nn_f


def masked_cross_entropy(logits, targets, mask):
    log_probabilities = log_softmax(logits, dim=1)
    losses = -torch.gather(log_probabilities, dim=1, index=targets.unsqueeze(1))
    if mask is None:
        loss_value = losses.mean()
    else:
        loss_value = torch.masked_select(losses, mask.bool()).mean()
    return loss_value


def masked_distance_loss(logits, target, mask, cell_size, indices):
    h_target = torch.floor(target / cell_size)
    w_target = target - h_target * cell_size
    probabilities = torch.softmax(logits, dim=1)
    _, prediction_index = torch.max(probabilities, dim=1)

    h_prediction = torch.floor(prediction_index / cell_size)
    w_prediction = prediction_index - h_prediction * cell_size
    h_diff = h_target - h_prediction
    w_diff = w_target - w_prediction
    losses = (h_diff * h_diff) + (w_diff * w_diff)
    losses /= cell_size * cell_size

    log_probabilities = log_softmax(logits, dim=1)
    classification_losses = -torch.gather(log_probabilities, dim=1, index=target.unsqueeze(1))

    losses = torch.where(target >= 64., classification_losses, losses)

    if mask is None:
        loss_value = losses.mean()
    else:
        loss_value = torch.masked_select(losses, mask.bool()).mean()

    return loss_value


class DetectorLoss(object):
    def __init__(self, is_cuda, cell_size):
        # self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.cell_size = cell_size
        self.indices = torch.arange(start=0, end=65)
        self.indices.unsqueeze_(dim=1)
        self.indices.unsqueeze_(dim=1)
        if is_cuda:
            self.indices = self.indices.cuda()

    def forward(self, points, true_points, valid_mask):
        if (valid_mask is not None) and len(valid_mask.shape) <= 2:  # skip empty masks
            valid_mask = None
        # was used for testing
        # loss_value = self.cross_entropy(points, true_points)
        # masked_loss_value = masked_cross_entropy(points, true_points, valid_mask)
        masked_loss_value = masked_distance_loss(points, true_points, valid_mask, self.cell_size, self.indices)

        return masked_loss_value

    def __call__(self, points, true_points, valid_mask):
        return self.forward(points, true_points, valid_mask)


class GlobalLoss(object):
    def __init__(self, settings):
        self.settings = settings
        self.detector_loss = DetectorLoss(settings.cuda, settings.cell)

    def forward(self, points,
                true_points,
                warped_points,
                warped_true_points,
                descriptors,
                warped_descriptors,
                homographies,
                valid_mask):
        # Compute the losses for the detector head
        detector_loss_value = self.detector_loss.forward(points, true_points, valid_mask=None)
        warped_detector_loss_value = self.detector_loss.forward(warped_points, warped_true_points, valid_mask)

        # Compute the loss for the descriptor head
        # this loss also maximize distance between non corresponding descriptors
        descriptor_loss_value = self.descriptor_loss(
            descriptors, warped_descriptors, homographies, valid_mask, self.settings.cell,
            lambda_d=self.settings.lambda_d, positive_margin=self.settings.positive_margin,
            negative_margin=self.settings.negative_margin)

        # descriptor_loss_value = self.descriptor_distance_loss(descriptors, warped_descriptors, homographies,
        #                                                       self.settings.cell, valid_mask)

        # loss = (detector_loss_value + warped_detector_loss_value) + descriptor_loss_value
        return [detector_loss_value, warped_detector_loss_value, descriptor_loss_value]

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

    def descriptor_distance_loss(self, descriptors, warped_descriptors, homographies, cell_size, valid_mask):
        # Compute the position of the center pixel of every cell in the image
        batch_size = descriptors.shape[0]
        hc = descriptors.shape[2]
        wc = descriptors.shape[3]

        # compare only similar descriptors - use homography transform to find correspondences
        coord_cells = torch.stack(torch.meshgrid(torch.arange(0, hc), torch.arange(0, wc)), dim=-1)
        coord_cells = coord_cells.to(device=descriptors.device)
        coord_cells = coord_cells * cell_size + cell_size // 2

        # resulting cells after mask filtration
        coord_cells = coord_cells.view([-1, 2])

        # Compute the positions of the warped cells
        warped_coord_cells = warp_points(coord_cells, homographies)
        coord_cells = coord_cells.repeat(batch_size, 1, 1)

        # Filter points laying out of the image shape
        shape_tensor = torch.tensor([hc * cell_size, wc * cell_size], dtype=torch.float, device=descriptors.device) - 1
        warp_mask = (warped_coord_cells < 0) | (warped_coord_cells > shape_tensor)
        warp_mask = torch.any(warp_mask, dim=2)
        outlier_num = warp_mask.sum()

        # restore cell coords
        coord_cells = (coord_cells - cell_size // 2) / cell_size
        warped_coord_cells = (warped_coord_cells - cell_size // 2) / cell_size

        # [0,0] - cell descriptor will be used for outliers
        coord_cells[warp_mask] = 0
        warped_coord_cells[warp_mask] = 0
        coord_cells = coord_cells.to(dtype=torch.long)
        warped_coord_cells = warped_coord_cells.to(dtype=torch.long)

        # Select correspondent descriptors
        batch_index = torch.arange(batch_size, dtype=torch.long)[:, None]
        descriptors = descriptors[batch_index, :, coord_cells[:, :, 0], coord_cells[:, :, 1]]
        warped_descriptors = warped_descriptors[batch_index, :, warped_coord_cells[:, :, 0], warped_coord_cells[:, :, 1]]

        # calculate correction for outliers
        descriptors[warp_mask] = 0
        warped_descriptors[warp_mask] = 0

        # Compute the Cosine Similarity loss
        # loss = nn_f.cosine_similarity(descriptors, warped_descriptors, dim=2)
        # loss = nn_f.cosine_embedding_loss(descriptors, warped_descriptors)

        # Normalize the descriptors
        # descriptors = normalize(descriptors, dim=1, p=2)
        # warped_descriptors = normalize(warped_descriptors, dim=1, p=2)

        # Compute the MSE loss
        loss = (descriptors - warped_descriptors) ** 2
        loss = loss.sum() / (torch.numel(loss) - outlier_num)

        return loss

    def descriptor_loss(self, descriptors, warped_descriptors, homographies, valid_mask, cell_size, positive_margin,
                        negative_margin, lambda_d):
        # Compute the position of the center pixel of every cell in the image
        batch_size = descriptors.shape[0]
        Hc = descriptors.shape[2]
        Wc = descriptors.shape[3]

        # Normalize the descriptors and
        # compute the pairwise dot product between descriptors: d^t * d'
        descriptors = torch.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
        descriptors = normalize(descriptors, dim=-1, p=2)
        warped_descriptors = torch.reshape(warped_descriptors,
                                           [batch_size, 1, 1, Hc, Wc, -1])
        warped_descriptors = normalize(warped_descriptors, dim=-1, p=2)

        # Original version
        dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=-1)

        # change precision
        # orig_type = descriptors.dtype
        # descriptors = descriptors.to(dtype=torch.float16)
        # warped_descriptors = warped_descriptors.to(dtype=torch.float16)
        # dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=-1)
        # dot_product_desc = dot_product_desc.to(dtype=orig_type)

        # use CPU
        # descriptors_cpu = descriptors.cpu()
        # warped_descriptors_cpu = warped_descriptors.cpu()
        # dot_product_desc = torch.sum(descriptors_cpu * warped_descriptors_cpu, dim=-1)
        # dot_product_desc = dot_product_desc.to(device=descriptors.device)

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

        zero = torch.tensor([0.], device=dot_product_desc.device)
        zero.expand_as(dot_product_desc)
        positive_dist = torch.maximum(zero, positive_margin - dot_product_desc)
        negative_dist = torch.maximum(zero, dot_product_desc - negative_margin)

        # cells
        coord_cells = torch.stack(torch.meshgrid(torch.arange(0, Hc), torch.arange(0, Wc)), dim=-1)
        coord_cells = coord_cells.to(device=positive_dist.device)
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

        # Compute the loss
        loss = lambda_d * s * positive_dist + (1 - s) * negative_dist

        # Mask the pixels if bordering artifacts appear
        valid_mask = torch.ones([batch_size, 1, 1, Hc, Wc], dtype=torch.float32) if valid_mask is None else valid_mask
        valid_mask = torch.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

        normalization = torch.sum(valid_mask) * float(Hc * Wc)
        loss = torch.sum(valid_mask * loss) / normalization

        return loss
