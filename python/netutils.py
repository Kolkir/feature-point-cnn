import numpy as np
import torch
from nms import corners_nms


def make_points_labels(points, img_h, img_w, cell_size):
    points_map = np.zeros((img_h, img_w))
    ys = points[:, 0].astype(int)
    xs = points[:, 1].astype(int)

    points_map[ys, xs] = 1

    img_h_cells = int(img_h / cell_size)
    img_w_cells = int(img_w / cell_size)

    points_map = np.reshape(points_map, [img_h_cells, img_w_cells, cell_size, cell_size])
    points_map = np.reshape(points_map, [img_h_cells, img_w_cells, cell_size * cell_size])
    # add a dustbin
    pad = ((0, 0), (0, 0), (0, 1))
    points_map = np.pad(points_map, pad_width=pad, mode='constant', constant_values=0)

    # Convert to labels - indices 0 - 65
    # Add a small random matrix to randomly break ties in argmax
    labels = np.argmax(points_map + np.random.uniform(0.0, 0.1, points_map.shape),
                       axis=2)

    return labels


def get_points(pointness_map, img_h, img_w, settings):
    pointness_map = pointness_map.data.cpu().numpy().squeeze()
    softmax_result = np.exp(pointness_map)
    softmax_result = softmax_result / (np.sum(softmax_result, axis=0) + .00001)
    # removing dustbin dimension
    no_dustbin = softmax_result[:-1, :, :]
    # reshape to get full resolution
    no_dustbin = no_dustbin.transpose(1, 2, 0)
    img_h_cells = int(img_h / settings.cell)
    img_w_cells = int(img_w / settings.cell)
    confidence_map = np.reshape(no_dustbin, [img_h_cells, img_w_cells, settings.cell, settings.cell])
    confidence_map = np.transpose(confidence_map, [0, 2, 1, 3])
    confidence_map = np.reshape(confidence_map,
                                [img_h_cells * settings.cell, img_w_cells * settings.cell])
    # threshold confidence level
    xs, ys = np.where(confidence_map >= settings.confidence_thresh)
    # if we didn't find any features
    if len(xs) == 0:
        return np.zeros((3, 0))

    # get points coordinates
    points = np.zeros((3, len(xs)))
    points[0, :] = ys
    points[1, :] = xs
    points[2, :] = confidence_map[xs, ys]
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
