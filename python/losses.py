import torch
import numpy as np
from torch.nn.functional import log_softmax


def masked_cross_entropy(logits, targets, class_weights, mask):
    log_probabilities = log_softmax(logits, dim=1)
    losses = -torch.gather(log_probabilities, dim=1, index=targets.unsqueeze(1))
    # weights
    weights = class_weights.view(-1, 1, 1)
    weights = weights.expand(-1, logits.size(2), logits.size(3))
    weights = torch.gather(weights, dim=0, index=targets)
    losses = losses * weights.unsqueeze(1)
    # apply mask
    if mask:  # skip empty masks
        losses = losses * mask.float()
        losses_flat = losses.flatten(start_dim=1)
        loss = torch.sum(losses_flat, dim=1)
        loss = loss / torch.count_nonzero(mask, dim=1)
    else:
        losses_flat = losses.flatten(start_dim=1)
        loss = torch.mean(losses_flat, dim=1)

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


def descriptor_loss(descriptors, warped_descriptors, homographies, valid_mask):
    return 10


#     # Compute the position of the center pixel of every cell in the image
#     (batch_size, Hc, Wc) = tf.unstack(tf.to_int32(tf.shape(descriptors)[:3]))
#     coord_cells = tf.stack(tf.meshgrid(
#         tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
#     coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
#     # coord_cells is now a grid containing the coordinates of the Hc x Wc
#     # center pixels of the 8x8 cells of the image
#
#     # Compute the position of the warped center pixels
#     warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
#     # warped_coord_cells is now a list of the warped coordinates of all the center
#     # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)
#
#     # Compute the pairwise distances and filter the ones less than a threshold
#     # The distance is just the pairwise norm of the difference of the two grids
#     # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
#     coord_cells = tf.to_float(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]))
#     warped_coord_cells = tf.reshape(warped_coord_cells,
#                                     [batch_size, Hc, Wc, 1, 1, 2])
#     cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
#     s = tf.to_float(tf.less_equal(cell_distances, config['grid_size'] - 0.5))
#     # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
#     # homography is at a distance from (h', w') less than config['grid_size']
#     # and 0 otherwise
#
#     # Normalize the descriptors and
#     # compute the pairwise dot product between descriptors: d^t * d'
#     descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
#     descriptors = tf.nn.l2_normalize(descriptors, -1)
#     warped_descriptors = tf.reshape(warped_descriptors,
#                                     [batch_size, 1, 1, Hc, Wc, -1])
#     warped_descriptors = tf.nn.l2_normalize(warped_descriptors, -1)
#     dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
#     dot_product_desc = tf.nn.relu(dot_product_desc)
#     dot_product_desc = tf.reshape(tf.nn.l2_normalize(
#         tf.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
#         3), [batch_size, Hc, Wc, Hc, Wc])
#     dot_product_desc = tf.reshape(tf.nn.l2_normalize(
#         tf.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
#         1), [batch_size, Hc, Wc, Hc, Wc])
#     # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
#     # descriptor at position (h, w) in the original descriptors map and the
#     # descriptor at position (h', w') in the warped image
#
#     # Compute the loss
#     positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
#     negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
#     loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist
#
#     # Mask the pixels if bordering artifacts appear
#     valid_mask = tf.ones([batch_size,
#                           Hc * config['grid_size'],
#                           Wc * config['grid_size']], tf.float32)\
#         if valid_mask is None else valid_mask
#     valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
#     valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
#     valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
#     valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])
#
#     normalization = tf.reduce_sum(valid_mask) * tf.to_float(Hc * Wc)
#     # Summaries for debugging
#     # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
#     # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
#     tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
#                                                      s * positive_dist) / normalization)
#     tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
#                                                      negative_dist) / normalization)
#     loss = tf.reduce_sum(valid_mask * loss) / normalization
#     return loss


class GlobalLoss(object):
    def __init__(self, is_cuda, lambda_loss):
        self.detector_loss = DetectorLoss(is_cuda)
        self.lambda_loss = lambda_loss

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
            descriptors, warped_descriptors, homographies, valid_mask
        )

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
