#include "superpoint.h"
#include <torch/linalg.h>
#include <torch/nn/functional.h>

namespace superpoint {
SuperPoint::SuperPoint(const std::string& script_file_name) {
  module_ = torch::jit::load(script_file_name);
}

std::vector<FeaturePoint> SuperPoint::ProcessFrame(const cv::Mat frame) {
  auto input = MatToTensor(frame);
  // swap axis
  input = input.permute({(2), (0), (1)});
  // add batch dim
  input.unsqueeze_(0);
  std::vector<torch::jit::IValue> inputs = {input};
  auto output = module_.forward(inputs);
  auto out_tuple = output.toTuple();
  auto pointness_map = out_tuple->elements()[0].toTensor();
  auto descriptors_map = out_tuple->elements()[1].toTensor();

  auto img_h = frame.rows;
  auto img_w = frame.cols;
  GetPoints(pointness_map, img_h, img_w);
  AddDescriptors(descriptors_map, img_h, img_w);
  return feature_points_;
}

void SuperPoint::AddDescriptors(at::Tensor descriptors_map,
                                int img_h,
                                int img_w) {
  // put feature points coordinates into a tensor
  sample_points_data_.resize(feature_points_.size() * 2);
  size_t index = 0;
  for (auto& fp : feature_points_) {
    sample_points_data_[index] = fp.x;
    sample_points_data_[index + feature_points_.size()] = fp.y;
    ++index;
  }
  at::Tensor sample_points = torch::from_blob(
      sample_points_data_.data(),
      {2, static_cast<long>(feature_points_.size())}, at::kFloat);

  // interpolate into descriptor map using 2D point locations
  using namespace torch::indexing;
  sample_points.index_put_(
      {0, "..."}, sample_points.index({0, "..."}) / (float(img_w) / 2.) - 1.);
  sample_points.index_put_(
      {1, "..."}, sample_points.index({1, "..."}) / (float(img_h) / 2.) - 1.);
  sample_points = sample_points.transpose(0, 1).contiguous();
  sample_points = sample_points.view({1, 1, -1, 2});
  if (descriptors_map.is_cuda())
    sample_points = sample_points.cuda();
  auto descriptors = torch::nn::functional::grid_sample(
      descriptors_map, sample_points,
      torch::nn::functional::GridSampleFuncOptions().align_corners(true));
  descriptors = descriptors.reshape({descriptors_map.size(1), -1});
  auto norm = torch::linalg::linalg_norm(
      descriptors, c10::nullopt, c10::IntArrayRef({0}), false, at::kFloat);
  descriptors /= norm;
  // move descriptors to cpu
  desc_map_data_.resize(descriptors.numel());

  torch::Tensor cpu_tensor =
      torch::from_blob(desc_map_data_.data(), descriptors.sizes(),
                       torch::TensorOptions(torch::kCPU).dtype(at::kFloat));
  cpu_tensor.copy_(descriptors);

  auto descriptor_len = descriptors.size(0);
  index = 0;
  for (auto& fp : feature_points_) {
    assert(static_cast<long>(fp.descriptor.size()) == descriptor_len);
    for (long di = 0; di < descriptor_len; ++di) {
      auto elem_index =
          index * descriptors.stride(1) + di * descriptors.stride(0);
      fp.descriptor[di] = desc_map_data_[elem_index];
    }
    ++index;
  }
}

void SuperPoint::GetPoints(const at::Tensor pointness_map,
                           int img_h,
                           int img_w) {
  using namespace torch::indexing;
  pointness_map.squeeze_();
  pointness_map.exp_();
  auto softmax_result = pointness_map / (pointness_map.sum({0}) + .00001);
  // removing dustbin dimension
  at::Tensor no_dustbin = softmax_result.index({Slice(None, -1), "...", "..."});
  // restore the full resolution:
  no_dustbin = no_dustbin.permute({1, 2, 0});

  auto img_h_cells = img_h / settings_.cell;
  auto img_w_cells = img_w / settings_.cell;

  at::Tensor confidence_map = no_dustbin.view(
      {img_h_cells, img_w_cells, settings_.cell, settings_.cell});
  confidence_map = confidence_map.permute({0, 2, 1, 3});
  confidence_map = confidence_map.reshape(
      {img_h_cells * settings_.cell, img_w_cells * settings_.cell});
  // threshold confidence level
  auto indices = at::where(confidence_map >= settings_.confidence_thresh);
  if (indices.empty()) {
    feature_points_.clear();
    return;
  }
  assert(indices.size() == 2);
  // get points coordinates
  at::Tensor xs = indices[0];
  at::Tensor ys = indices[1];
  auto points_num = xs.size(0);
  auto points = torch::zeros({3, points_num}, at::kFloat);
  points.index_put_({0, "..."}, xs);
  points.index_put_({1, "..."}, ys);
  points.index_put_({2, "..."}, confidence_map.index({xs, ys}));
  // NMS
  FeatureNMS(points, img_h, img_w, settings_.nms_dist, settings_.border_remove,
             feature_points_, features_data_);
}
}  // namespace superpoint
