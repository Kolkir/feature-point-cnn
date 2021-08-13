import numpy as np
import torch
from src.nms import corners_nms


def scale_valid_map(data_map, img_h, img_w, cell_size):
    img_h_cells = int(img_h / cell_size)
    img_w_cells = int(img_w / cell_size)

    data_map = torch.reshape(data_map, [img_h_cells, cell_size, img_w_cells, cell_size])
    data_map = data_map.permute([0, 2, 1, 3])
    data_map = torch.reshape(data_map, [img_h_cells, img_w_cells, cell_size * cell_size])
    data_map = data_map.permute([2, 0, 1])

    data_map = torch.sum(data_map, dim=0)
    data_map = torch.where(data_map > 0., 1., 0.)

    return data_map.unsqueeze(dim=0)


def make_prob_map_from_labels(labels, img_h, img_w, cell_size):
    prob_map = np.zeros((labels.shape[0], labels.shape[1], 65))
    indices = np.argwhere(labels < 64)
    depth_indices = labels[indices[:, 0], indices[:, 1]]
    prob_map[indices[:, 0], indices[:, 1], depth_indices] = 1
    prob_map = prob_map.transpose([2, 0, 1])
    prob_map = restore_prob_map(torch.from_numpy(prob_map).unsqueeze(dim=0), img_h, img_w, cell_size)
    return prob_map


def make_points_labels(points, img_h, img_w, cell_size):
    points_map = np.zeros((img_h, img_w))
    ys = points[:, 0].astype(int)
    xs = points[:, 1].astype(int)
    points_map[ys, xs] = 2  # assign a highest score to where the corners are

    img_h_cells = int(img_h / cell_size)
    img_w_cells = int(img_w / cell_size)

    points_map = np.reshape(points_map, [img_h_cells, cell_size, img_w_cells, cell_size])
    points_map = np.transpose(points_map, [0, 2, 1, 3])
    points_map = np.reshape(points_map, [img_h_cells, img_w_cells, cell_size * cell_size])
    # add a dustbin, and assign the second level score to be bigger than noise
    pad = ((0, 0), (0, 0), (0, 1))
    points_map = np.pad(points_map, pad_width=pad, mode='constant', constant_values=1)
    points_map = points_map.transpose([2, 0, 1])

    # Convert to labels - indices 0 - 65
    # Add a small random matrix to randomly break ties in argmax - if the 8x8 region has several corners
    labels = np.argmax(points_map + np.random.uniform(0.0, 0.1, points_map.shape),
                       axis=0)

    return labels


def get_points_coordinates(prob_map, img_h, img_w, cell_size, confidence_thresh):
    prob_map = prob_map.data.cpu().numpy()
    # threshold confidence level
    bs, ys, xs = np.where(prob_map >= confidence_thresh)
    confidence = prob_map[bs, ys, xs]
    return ys, xs, confidence


def restore_prob_map(prob_map, img_h, img_w, cell_size):
    # removing dustbin dimension
    no_dustbin = prob_map[:, :-1, :, :]
    # reshape to get full resolution
    img_h_cells = int(img_h / cell_size)
    img_w_cells = int(img_w / cell_size)
    no_dustbin = no_dustbin.permute([0, 2, 3, 1])
    confidence_map = torch.reshape(no_dustbin, [-1, img_h_cells, img_w_cells, cell_size, cell_size])
    confidence_map = confidence_map.permute([0, 1, 3, 2, 4])
    confidence_map = torch.reshape(confidence_map,
                                   [-1, img_h_cells * cell_size, img_w_cells * cell_size])
    return confidence_map


def get_points(prob_map, img_h, img_w, settings):
    ys, xs, confidence = get_points_coordinates(prob_map, img_h, img_w, settings.cell, settings.confidence_thresh)
    # if we didn't find any features
    if len(xs) == 0:
        return np.zeros((3, 0))

    # get points coordinates
    points = np.zeros((3, len(xs)))
    points[0, :] = xs
    points[1, :] = ys
    points[2, :] = confidence
    # NMS
    points = corners_nms(points, img_h, img_w, dist_thresh=settings.nms_dist)
    # sort by confidence(why do we need this? nms returns sorted values)
    indices = np.argsort(points[2, :])
    points = points[:, indices[::-1]]
    # remove points along border.
    border_width = settings.border_remove
    horizontal_remove_idx = np.logical_or(points[0, :] < border_width, points[0, :] >= (img_w - border_width))
    vertical_remove_idx = np.logical_or(points[1, :] < border_width, points[1, :] >= (img_h - border_width))
    total_remove_idx = np.logical_or(horizontal_remove_idx, vertical_remove_idx)
    points = points[:, ~total_remove_idx]
    return points


def get_descriptors(points, descriptors_map, img_h, img_w, settings):
    # dn = norm(desc, p=2, dim=1)
    # desc = desc.div(unsqueeze(dn, 1))

    if points.shape[1] == 0:
        return np.zeros((descriptors_map.shape[1], 0))
    # interpolate into descriptor map using 2D point locations
    sample_points = torch.from_numpy(points[:2, :].copy())
    sample_points[0, :] = (sample_points[0, :] / (float(img_w) / 2.)) - 1.
    sample_points[1, :] = (sample_points[1, :] / (float(img_h) / 2.)) - 1.
    sample_points = sample_points.transpose(0, 1).contiguous()
    sample_points = sample_points.view(1, 1, -1, 2)
    sample_points = sample_points.float()
    if settings.cuda:
        sample_points = sample_points.cuda()
    desc = torch.nn.functional.grid_sample(descriptors_map, sample_points, align_corners=True)
    desc = desc.data.cpu().numpy().reshape(descriptors_map.shape[1], -1)
    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc
