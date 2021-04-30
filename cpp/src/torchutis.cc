#include "torchutis.h"

namespace superpoint {

at::Tensor MatToTensor(cv::Mat img) {
  assert(img.type() == CV_32FC1);
  at::Tensor tensor_image =
      torch::from_blob(img.data, {img.rows, img.cols, 1}, at::kFloat);
  return tensor_image;
}
}  // namespace superpoint
