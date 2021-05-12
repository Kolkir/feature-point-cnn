#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "settings.h"
#include "torchutis.h"

namespace superpoint {

class SuperPoint {
 public:
  SuperPoint(const std::string& file_name, bool load_sript);
  SuperPoint(const SuperPoint&) = delete;
  SuperPoint& operator=(const SuperPoint&) = delete;

  std::vector<FeaturePoint> ProcessFrame(const cv::Mat& frame);

 private:
  void GetPoints(const at::Tensor pointness_map, int img_h, int img_w);

  void AddDescriptors(at::Tensor descriptors_map, int img_h, int img_w);

 private:
  Settings settings_;
  torch::jit::script::Module module_;
  SPModel model_{nullptr};
  bool use_script_ = true;

  // memory management buffers
  std::vector<float> desc_map_data_;
  std::vector<float> sample_points_data_;
  std::vector<FeaturePoint> feature_points_;
  std::vector<float> features_data_;
};

}  // namespace superpoint

#endif  // SUPERPOINT_H
