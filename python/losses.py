import torch
import numpy as np
from torch.nn.functional import log_softmax


def masked_cross_entropy(logits, targets, class_weights, mask):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probabilities_flat = log_softmax(logits_flat)
    targets_flat = targets.view(-1, 1)
    losses_flat = -torch.gather(log_probabilities_flat, dim=1, index=targets_flat)
    # apply weights
    weights_tensor = torch.ones(size=logits.shape)
    weights_tensor = weights_tensor * class_weights
    weights_tensor_flat = weights_tensor.view(-1, weights_tensor.size(-1))
    weights_tensor_flat = torch.gather(weights_tensor_flat, dim=1, index=targets_flat)
    losses_flat = losses_flat * weights_tensor_flat
    # restore dimensions
    losses = losses_flat.view(*targets.size())
    # apply mask
    losses = losses * mask.float()
    # average
    loss = losses.sum() / torch.count_nonzero(mask)

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
        # self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weight)

    def forward(self, points, true_points, valid_mask):
        # return self.cross_entropy(points, true_points)
        return masked_cross_entropy(points, true_points, self.class_weight, valid_mask)

    def __call__(self, points, true_points, descriptors, warped_descriptors, valid_mask):
        return self.forward(points, true_points, valid_mask)

# def descriptor_loss(descriptors, warped_descriptors, homographies,
#                     valid_mask=None, **config):
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


# def global_loss(self, outputs, inputs, **config):
#     logits = outputs['logits']
#     warped_logits = outputs['warped_results']['logits']
#     descriptors = outputs['descriptors_raw']
#     warped_descriptors = outputs['warped_results']['descriptors_raw']
#
#     # Switch to 'channels last' once and for all
#     if config['data_format'] == 'channels_first':
#         logits = tf.transpose(logits, [0, 2, 3, 1])
#         warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
#         descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
#         warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])
#
#     # Compute the loss for the detector head
#     detector_loss = utils.detector_loss(
#         inputs['keypoint_map'], logits,
#         valid_mask=inputs['valid_mask'], **config)
#     warped_detector_loss = utils.detector_loss(
#         inputs['warped']['keypoint_map'], warped_logits,
#         valid_mask=inputs['warped']['valid_mask'], **config)
#
#     # Compute the loss for the descriptor head
#     descriptor_loss = utils.descriptor_loss(
#         descriptors, warped_descriptors, outputs['homography'],
#         valid_mask=inputs['warped']['valid_mask'], **config)
#
#     tf.summary.scalar('detector_loss1', detector_loss)
#     tf.summary.scalar('detector_loss2', warped_detector_loss)
#     tf.summary.scalar('detector_loss_full', detector_loss + warped_detector_loss)
#     tf.summary.scalar('descriptor_loss', config['lambda_loss'] * descriptor_loss)
#
#     loss = (detector_loss + warped_detector_loss
#             + config['lambda_loss'] * descriptor_loss)
#     return loss
