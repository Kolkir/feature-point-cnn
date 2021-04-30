#include <torch/script.h>
#include <iostream>
#include "CLI11.hpp"
#include "camera.h"
#include "torchutis.h"

int main(int argc, char* argv[]) {
  CLI::App app{"SuperPoint NN Cpp demo"};

  std::string script_filename = "superpoint.pt";
  app.add_option("-f,--script_file", script_filename,
                 "A path to the NN script file");

  int width = 640;
  app.add_option("--width", width, "Camera frame resize width");

  int height = 480;
  app.add_option("--height", height, "Camera frame resize height");

  int cam_index = 0;
  app.add_option("-i,--cam_id", cam_index, "Camera index");

  CLI11_PARSE(app, argc, argv);

  try {
    torch::jit::script::Module module = torch::jit::load(script_filename);
    std::cout << "Model loaded\n";
    superpoint::Camera camera(cam_index, width, height);
    std::cout << "Camera initialized\n";

    std::string win_name("SuperPoint Cpp demo");
    cv::namedWindow(win_name);

    bool done = false;
    while (!done) {
      cv::Mat frame = camera.get_frame();
      auto input = superpoint::MatToTensor(frame);
      // swap axis
      input = input.permute({(2), (0), (1)});
      // add batch dim
      input.unsqueeze_(0);
      std::vector<torch::jit::IValue> inputs = {input};
      auto output = module.forward(inputs);
      auto out_tuple = output.toTuple();
      auto pointness = out_tuple->elements()[0].toTensor();
      auto descriptors = out_tuple->elements()[1].toTensor();

      cv::imshow(win_name, frame);
      int key = cv::waitKey(1);
      if (key == 'q') {
        done = true;
      }
    }
    cv::destroyAllWindows();
  } catch (const std::exception& e) {
    std::cerr << "Processing error: \n" << e.what() << std::endl;
    return -1;
  }

  return 0;
}
