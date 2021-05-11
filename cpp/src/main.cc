#include <chrono>
#include <iostream>
#include "CLI11.hpp"
#include "camera.h"
#include "superpoint.h"

using namespace superpoint;

double DescriptorDist(const DescriptorType& a, const DescriptorType& b) {
  assert(a.size() == b.size());
  double sum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return sqrt(sum);
}

const FeaturePoint* SearchKeyFrameCorrespondence(
    const std::vector<FeaturePoint>& key_frame,
    const FeaturePoint& fp,
    double tollerance) {
  for (auto& key_fp : key_frame) {
    auto dist = DescriptorDist(key_fp.descriptor, fp.descriptor);
    if (dist < tollerance) {
      return &key_fp;
    }
  }
  return nullptr;
}

int main(int argc, char* argv[]) {
  CLI::App app{
      "SuperPoint NN Cpp demo, press q to quit and k to save key-frame"};

  std::string script_filename = "superpoint.pt";
  app.add_option("-f,--script_file", script_filename,
                 "A path to the NN script file")
      ->required();

  int width = 640;
  app.add_option("--width", width, "Camera frame resize width");

  int height = 480;
  app.add_option("--height", height, "Camera frame resize height");

  int cam_index = 0;
  app.add_option("-i,--cam_id", cam_index, "Camera index");

  CLI11_PARSE(app, argc, argv);

  const double desc_tollerance = 0.8;
  const size_t num_key_features = 20;
  std::vector<FeaturePoint> key_frame;
  key_frame.reserve(num_key_features);

  try {
    superpoint::SuperPoint net(script_filename);
    std::cout << "Model loaded\n";
    superpoint::Camera camera(cam_index, width, height);
    std::cout << "Camera initialized\n";

    std::string win_name("SuperPoint Cpp demo");
    cv::namedWindow(win_name);

    bool done = false;
    cv::Mat key_frame_img;
    cv::Mat comb_frame;
    std::vector<FeaturePoint> feature_points;
    auto begin_frame_time = std::chrono::steady_clock::now();
    size_t fps_num = 0;
    size_t last_fps_num = 0;
    std::string fps_label = "FPS: ";
    while (!done) {
      auto [frame, frame_gray] = camera.GetFrame();
      feature_points = net.ProcessFrame(frame_gray);

      size_t index = 0;
      for (const auto& key_fp : key_frame) {
        const auto* fp = SearchKeyFrameCorrespondence(feature_points, key_fp,
                                                      desc_tollerance);
        if (fp) {
          auto label = std::to_string(index);
          cv::putText(frame, label, cv::Point(fp->x, fp->y),
                      cv::FONT_HERSHEY_COMPLEX_SMALL,
                      /*font scale*/ 1, cv::Scalar(0, 0, 250),
                      /*thickness*/ 1,
                      /*line type*/ cv::LINE_AA);
        }
        ++index;
      }

      if (key_frame.empty()) {
        comb_frame = frame;
      } else {
        cv::Mat keyRoi =
            comb_frame(cv::Rect(0, 0, key_frame_img.cols, key_frame_img.rows));
        key_frame_img.copyTo(keyRoi);
        cv::Mat frameRoi = comb_frame(cv::Rect(
            key_frame_img.cols, 0, key_frame_img.cols, key_frame_img.rows));
        frame.copyTo(frameRoi);
      }
      // show FPS
      fps_label.resize(5);
      fps_label += std::to_string(last_fps_num);
      cv::putText(comb_frame, fps_label, cv::Point(10, 30),
                  cv::FONT_HERSHEY_COMPLEX_SMALL,
                  /*font scale*/ 1, cv::Scalar(255, 255, 255),
                  /*thickness*/ 1,
                  /*line type*/ cv::LINE_AA);

      cv::imshow(win_name, comb_frame);
      int key = cv::waitKey(1);
      if (key == 'q') {
        done = true;
      }
      if (key == 'k') {
        comb_frame = cv::Mat(frame.rows, frame.cols * 2, frame.type());
        key_frame_img = frame.clone();
        key_frame.resize(std::min(num_key_features, feature_points.size()));
        std::copy_n(feature_points.begin(), key_frame.size(),
                    key_frame.begin());

        size_t index = 0;
        for (const auto& fp : key_frame) {
          auto label = std::to_string(index);
          cv::putText(key_frame_img, label, cv::Point(fp.x, fp.y),
                      cv::FONT_HERSHEY_COMPLEX_SMALL,
                      /*font scale*/ 1, cv::Scalar(0, 250, 0),
                      /*thickness*/ 1,
                      /*line type*/ cv::LINE_AA);

          ++index;
        }
      }
      auto now = std::chrono::steady_clock::now();
      if (now - begin_frame_time >= std::chrono::seconds{1}) {
        last_fps_num = fps_num;
        fps_num = 0;
        begin_frame_time = now;
      }
      ++fps_num;
    }
    cv::destroyAllWindows();
  } catch (const std::exception& e) {
    std::cerr << "Processing error: \n" << e.what() << std::endl;
    return -1;
  }

  return 0;
}
