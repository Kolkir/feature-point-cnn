#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "settings.h"

namespace superpoint {
class SPModelImpl : public torch::nn::Module {
 public:
  SPModelImpl(const Settings& settings);
  SPModelImpl(const SPModelImpl&) = delete;
  SPModelImpl& operator=(const SPModelImpl&) = delete;

  c10::IValue forward(torch::Tensor input);

 private:
  std::vector<torch::nn::Conv2d> encoder_conv_;

  torch::nn::Conv2d detector_conv_a_{nullptr};
  torch::nn::Conv2d detector_conv_b_{nullptr};

  torch::nn::Conv2d descriptor_conv_a_{nullptr};
  torch::nn::Conv2d descriptor_conv_b_{nullptr};
};

TORCH_MODULE(SPModel);
}  // namespace superpoint

#endif  // MODEL_H
