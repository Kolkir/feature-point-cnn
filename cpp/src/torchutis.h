#ifndef TORCHUTIS_H
#define TORCHUTIS_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace superpoint {

at::Tensor MatToTensor(cv::Mat img);

using DescriptorType = std::array<float, 256>;

struct FeaturePoint {
  int x = 0;
  int y = 0;
  float confidence = 0;
  DescriptorType descriptor;
};

void FeatureNMS(at::Tensor feature_points_tensor,
                int img_h,
                int img_w,
                int nms_dist,
                int border_width,
                std::vector<FeaturePoint>& feature_points,
                std::vector<float>& features_data);

}  // namespace superpoint

#endif  // TORCHUTIS_H
