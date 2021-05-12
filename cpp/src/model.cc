#include "model.h"

namespace superpoint {
SPModelImpl::SPModelImpl(const Settings& settings) {
  // Create encoder
  int i = 0;
  for (auto dim : settings.encoder_dims) {
    auto conv_a = register_module(
        "encoder_conv" + std::to_string(i) + "_a",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(dim.first, dim.second,
                                                   settings.encoder_kernel_size)
                              .stride(settings.encoder_stride)
                              .padding(settings.encoder_padding)));
    encoder_conv_.push_back(std::move(conv_a));

    auto conv_b = register_module(
        "encoder_conv" + std::to_string(i) + "_b",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(dim.second, dim.second,
                                                   settings.encoder_kernel_size)
                              .stride(settings.encoder_stride)
                              .padding(settings.encoder_padding)));

    encoder_conv_.push_back(std::move(conv_b));
    ++i;
  }

  // Create detector head
  detector_conv_a_ = register_module(
      "detector_conv_a",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(settings.detector_dims[0],
                                                 settings.detector_dims[1],
                                                 settings.detdesc_kernel_size_a)
                            .stride(settings.detdesc_stride)
                            .padding(settings.detdesc_padding_a)));
  detector_conv_b_ = register_module(
      "detector_conv_b",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(settings.detector_dims[1],
                                                 settings.detector_dims[2],
                                                 settings.detdesc_kernel_size_b)
                            .stride(settings.detdesc_stride)
                            .padding(settings.detdesc_padding_b)));

  // Create descriptor head
  descriptor_conv_a_ = register_module(
      "descriptor_conv_a",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(settings.descriptor_dims[0],
                                                 settings.descriptor_dims[1],
                                                 settings.detdesc_kernel_size_a)
                            .stride(settings.detdesc_stride)
                            .padding(settings.detdesc_padding_a)));
  descriptor_conv_b_ = register_module(
      "descriptor_conv_b",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(settings.descriptor_dims[1],
                                                 settings.descriptor_dims[2],
                                                 settings.detdesc_kernel_size_b)
                            .stride(settings.detdesc_stride)
                            .padding(settings.detdesc_padding_b)));
}

std::pair<at::Tensor, at::Tensor> SPModelImpl::forward(torch::Tensor input) {
  namespace tf = torch::nn::functional;
  const auto relu_opt = tf::ReLUFuncOptions().inplace(true);
  at::Tensor x = input;
  // encoder
  size_t last_step = encoder_conv_.size() - 2;
  for (size_t i = 0; i < encoder_conv_.size(); i += 2) {
    x = encoder_conv_[i](x);
    tf::relu(x, relu_opt);
    // x = torch::relu(x);
    x = encoder_conv_[i + 1](x);
    tf::relu(x, relu_opt);
    // x = torch::relu(x);
    if (i != last_step) {
      x = torch::max_pool2d(x, /*kernel_size*/ 2, /*stride*/ 2);
    }
  }

  // detector head
  auto point = detector_conv_a_(x);
  tf::relu(point, relu_opt);
  // point = torch::relu(point);
  point = detector_conv_b_(point);

  // descriptor head
  auto desc = descriptor_conv_a_(x);
  tf::relu(desc, relu_opt);
  // desc = torch::relu(desc);
  desc = descriptor_conv_b_(desc);

  auto dn = torch::norm(desc, /*p*/ 2, /*dim*/ 1);
  desc = desc.div(torch::unsqueeze(dn, 1));  // normalize

  return {point, desc};
}
}  // namespace superpoint
