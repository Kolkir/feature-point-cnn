#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

namespace superpoint {
class Camera {
 public:
  Camera(int index, int width, int height);
  std::pair<cv::Mat, cv::Mat> GetFrame();

 private:
  cv::VideoCapture cap_;
  int resize_width_ = 640;
  int resize_height_ = 480;
};
}  // namespace superpoint

#endif  // CAMERA_H
