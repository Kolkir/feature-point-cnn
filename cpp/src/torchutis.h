#ifndef TORCHUTIS_H
#define TORCHUTIS_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace superpoint {

at::Tensor MatToTensor(cv::Mat img);

}

#endif  // TORCHUTIS_H
