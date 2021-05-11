#include "torchutis.h"

namespace superpoint {

at::Tensor MatToTensor(cv::Mat img) {
  assert(img.type() == CV_32FC1);
  at::Tensor tensor_image =
      torch::from_blob(img.data, {img.rows, img.cols, 1}, at::kFloat);
  return tensor_image;
}

static void TensorToFeaturePoints(at::Tensor feature_points_tensor,
                                  std::vector<FeaturePoint>& feature_points,
                                  std::vector<float>& features_data) {
  long cols = 3;
  assert(feature_points_tensor.size(0) == cols);
  auto num_features = feature_points_tensor.size(1);

  features_data.resize(num_features * cols);
  torch::Tensor cpu_tensor =
      torch::from_blob(features_data.data(), feature_points_tensor.sizes(),
                       torch::TensorOptions(torch::kCPU).dtype(at::kFloat));
  cpu_tensor.copy_(feature_points_tensor);

  feature_points.resize(num_features);

  auto stride = num_features;
  for (long row = 0; row < num_features; ++row) {
    FeaturePoint& fp = feature_points[row];

    fp.y = static_cast<int>(features_data[row]);
    fp.x = static_cast<int>(features_data[row + stride]);
    fp.confidence = features_data[row + stride * 2];
  }
}

void FeatureNMS(at::Tensor feature_points_tensor,
                int img_h,
                int img_w,
                int nms_dist,
                int border_width,
                std::vector<FeaturePoint>& feature_points,
                std::vector<float>& features_data) {
  TensorToFeaturePoints(feature_points_tensor, feature_points, features_data);
  if (feature_points.size() > 1) {
    // sort feature points highest to lowest conf
    std::sort(feature_points.begin(), feature_points.end(),
              [](const FeaturePoint& a, const FeaturePoint& b) {
                return a.confidence < b.confidence;
              });
    std::reverse(feature_points.begin(), feature_points.end());

    auto pad = nms_dist;  // padding allows NMS points near the border
    // allocate grid
    std::vector<std::vector<int>> grid;
    grid.resize(img_h + pad * 2);
    for (auto& row : grid)
      row.resize(img_w + pad * 2, 0);
    // initialize grid
    for (const auto& fp : feature_points) {
      grid[fp.y + pad][fp.x + pad] = 1;
    }
    // iterate through points, highest to lowest conf, suppress neighborhood.
    for (const auto& fp : feature_points) {
      // account for top and left padding.
      int x = fp.x + pad;
      int y = fp.y + pad;
      // check in not yet suppressed.
      if (grid[y][x] == 1) {
        // suppress points by setting nearby values to 0.
        for (int sy = y - pad; sy <= y + pad; ++sy) {
          for (int sx = x - pad; sx <= x + pad; ++sx) {
            grid[sy][sx] = 0;
          }
        }
        // keep the point or remove it if it is too close to the border
        auto is_far_from_horisontal_border =
            fp.x > border_width || fp.x < (img_w - border_width);
        auto is_far_from_vertical_border =
            fp.y > border_width || fp.y < (img_h - border_width);

        if (is_far_from_horisontal_border && is_far_from_vertical_border) {
          grid[y][x] = -1;
        }
      }
    }
    // get all surviving -1's
    std::vector<FeaturePoint> out_fp;
    out_fp.reserve(feature_points.size());
    for (const auto& fp : feature_points) {
      // account for top and left padding.
      int x = fp.x + pad;
      int y = fp.y + pad;
      if (grid[y][x] == -1) {
        out_fp.push_back(fp);
      }
    }
  }
}
}  // namespace superpoint
