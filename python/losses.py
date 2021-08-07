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
from homographies import warp_points


def masked_cross_entropy(logits, targets, mask):
    log_probabilities = log_softmax(logits, dim=1)
    losses = -torch.gather(log_probabilities, dim=1, index=targets.unsqueeze(1))
    if mask is None:
        loss_value = losses.mean()
    else:
        loss_value = torch.masked_select(losses, mask.bool()).mean()
    return loss_value


class DetectorLoss(object):
    def __init__(self, is_cuda):
        # self.cross_entropy = torch.nn.CrossEntropyLoss()
        pass

    def forward(self, points, true_points, valid_mask):
        if (valid_mask is not None) and len(valid_mask.shape) <= 2:  # skip empty masks
            valid_mask = None
        # was used for testing
        # loss_value = self.cross_entropy(points, true_points)
        masked_loss_value = masked_cross_entropy(points, true_points, valid_mask)

        return masked_loss_value

    def __call__(self, points, true_points, valid_mask):
        return self.forward(points, true_points, valid_mask)


class GlobalLoss(object):
    def __init__(self, is_cuda, lambda_loss, settings):
        self.detector_loss = DetectorLoss(is_cuda)
        self.lambda_loss = lambda_loss
        self.cell_size = settings.cell
        self.detector_loss_value = None
        self.warped_detector_loss_value = None
        self.descriptor_loss_value = None

    def forward(self, points,
                true_points,
                warped_points,
                warped_true_points,
                descriptors,
                warped_descriptors,
                homographies,
                valid_mask):
        # Compute the losses for the detector head
        self.detector_loss_value = self.detector_loss.forward(points, true_points, valid_mask=None)
        self.warped_detector_loss_value = self.detector_loss.forward(
            warped_points, warped_true_points,
            valid_mask)

        # Compute the loss for the descriptor head
        self.descriptor_loss_value = self.descriptor_loss(
            descriptors, warped_descriptors, homographies, valid_mask, self.cell_size,
            lambda_d=250, positive_margin=1, negative_margin=0.2)

        loss = (self.detector_loss_value + self.warped_detector_loss_value
                + self.lambda_loss * self.descriptor_loss_value)

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
