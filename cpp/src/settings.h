#ifndef SETTINGS_H
#define SETTINGS_H

#include <torch/torch.h>
#include <vector>

namespace superpoint {
struct Settings {
  torch::IntArrayRef encoder_kernel_size = {3, 3};
  torch::IntArrayRef encoder_stride = {1, 1};
  torch::IntArrayRef encoder_padding = {1, 1};

  torch::IntArrayRef detdesc_kernel_size_a = {3, 3};
  torch::IntArrayRef detdesc_kernel_size_b = {1, 1};
  torch::IntArrayRef detdesc_stride = {1, 1};
  torch::IntArrayRef detdesc_padding_a = {1, 1};
  int64_t detdesc_padding_b = 0;

  std::vector<std::pair<int64_t, int64_t>> encoder_dims = {{1, 64},
                                                           {64, 64},
                                                           {64, 128},
                                                           {128, 128}};

  std::vector<int64_t> detector_dims = {128, 256, 65};
  std::vector<int64_t> descriptor_dims = {128, 256, 256};

  int nms_dist = 4;
  float confidence_thresh = 0.015f;
  float nn_thresh = 0.7f;  // L2 descriptor distance for good match.
  int cell = 8;            // Size of each output cell. Keep this fixed.
  int border_remove = 4;   // Remove points this close to the border.
};
}  // namespace superpoint
#endif  // SETTINGS_H
