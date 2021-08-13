import numpy as np


def corners_nms(in_corners, img_h, img_w, dist_thresh):
    """
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      corners - 3xN numpy matrix with surviving corners.
    """
    grid = np.zeros((img_h, img_w)).astype(int)
    indices = np.zeros((img_h, img_w)).astype(int)
    # Sort by confidence and round to nearest int.
    confidence_indices = np.argsort(-in_corners[2, :])
    corners = in_corners[:, confidence_indices]
    rounded_corners = corners[:2, :].round().astype(int)
    # Check for edge case of 0 or 1 corners.
    if rounded_corners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int)
    if rounded_corners.shape[1] == 1:
        out = np.vstack((rounded_corners, in_corners[2])).reshape(3, 1)
        return out
    # Initialize the grid.
    #   -1 : Kept.
    #    0 : Empty or suppressed.
    #    1 : To be processed (converted to either kept or supressed).
    for i, rc in enumerate(rounded_corners.T):
        grid[rounded_corners[1, i], rounded_corners[0, i]] = 1
        indices[rounded_corners[1, i], rounded_corners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    for i, rc in enumerate(rounded_corners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # check in not yet suppressed.
            # suppress points by setting nearby values to 0.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            # keep the point
            grid[pt[1], pt[0]] = -1
    # Get all surviving -1's and return sorted array of remaining corners.
    keep_y, keep_x = np.where(grid == -1)
    keep_y, keep_x = keep_y - pad, keep_x - pad
    indices_to_keep = indices[keep_y, keep_x]
    out = corners[:, indices_to_keep]
    values = out[-1, :]
    out_indices = np.argsort(-values)
    out = out[:, out_indices]
    return out
