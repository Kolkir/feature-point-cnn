#include "camera.h"

namespace superpoint {
Camera::Camera(int index, int width, int height)
    : resize_width_(width), resize_height_(height) {
  cap_ = cv::VideoCapture(index, cv::CAP_V4L2);
  if (!cap_.isOpened()) {
    throw std::runtime_error("Failed to open a camera");
  }
}

std::pair<cv::Mat, cv::Mat> Camera::GetFrame() {
  cv::Mat frame;
  cv::Mat frame_gray;
  if (cap_.read(frame)) {
    cv::resize(frame, frame_gray, cv::Size(resize_width_, resize_height_));
    cv::cvtColor(frame_gray, frame_gray, cv::COLOR_BGR2GRAY);
    frame_gray.convertTo(frame_gray, CV_32FC1, 1.0 / 255.0);
  } else {
    std::cerr << "Failed to read a camera frame" << std::endl;
  }
  return {frame, frame_gray};
}
}  // namespace superpoint
