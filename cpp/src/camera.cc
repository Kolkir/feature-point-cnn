#include "camera.h"

namespace superpoint {
Camera::Camera(int index, int width, int height)
    : resize_width_(width), resize_height_(height) {
  cap = cv::VideoCapture(index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open a camera");
  }
}

cv::Mat Camera::get_frame() {
  cv::Mat frame;
  if (cap.read(frame)) {
    cv::resize(frame, frame, cv::Size(resize_width_, resize_height_));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_32FC1, 1.0 / 255.0);
  } else {
    std::cerr << "Failed to read a camera frame" << std::endl;
  }
  return frame;
}
}  // namespace superpoint
