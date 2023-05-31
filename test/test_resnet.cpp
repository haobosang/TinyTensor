#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
#include "../source/layer/details/softmax.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
using namespace TinyTensor;

// python ref https://pytorch.org/hub/pytorch_vision_resnet/
std::shared_ptr<Tensor<float>> PreProcessImage(const cv::Mat& image) {

  assert(!image.empty());
  // 调整输入大小
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  rgb_image.convertTo(rgb_image, CV_32FC3);
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);
  uint32_t input_w = 224;
  uint32_t input_h = 224;
  uint32_t input_c = 3;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_c, input_h, input_w);

  uint32_t index = 0;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
  }

  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;
  assert(input->channels() == 3);
  input->data() = input->data() / 255.f;
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;
  return input;
}
TEST(test_initinoutput, init_init_graph) {
  using namespace TinyTensor;
  const std::string &param_path = "../../tmp/test.pnnx.param";
  const std::string &bin_path = "../../tmp/test.pnnx.bin";
  std::cout<<param_path;
  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  LOG(INFO) << "Start TinyTensor inference";
  std::shared_ptr<Tensor<float>> input1 =
      std::make_shared<Tensor<float>>(1,16,16);
  input1->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);


  std::vector<std::shared_ptr<Tensor<float>>> outputs =
      graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);
  
  // const auto &output1 = outputs.front()->shapes();
  // for(auto it:output1){
  //   std::cout<<it<<" ";
  // }
}

TEST(test_initinoutput, init_init_graph1) {
  using namespace TinyTensor;
  const std::string &param_path = "../../tmp/resnet18_batch1.pnnx.param";
  const std::string &bin_path = "../../tmp/resnet18_batch1.pnnx.bin";
  std::cout<<param_path;
  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  LOG(INFO) << "Start TinyTensor inference";
  std::shared_ptr<Tensor<float>> input1 =
      std::make_shared<Tensor<float>>(3,224,224);
  input1->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);


  std::vector<std::shared_ptr<Tensor<float>>> outputs =
      graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);
  
  const auto &output1 = outputs.front()->shapes();
  for(auto it:output1){
    std::cout<<it<<" ";
  }
}
 