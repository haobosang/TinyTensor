//#include "data/Tensor.hpp"
//#include "data/load_data.hpp"
#include<memory>
#include<gtest/gtest.h>
//#include<armadillo>
#include<glog/logging.h>
#include "data/Tensor.hpp"
#include "ops/sigmod_op.hpp"
#include "layer/sigmod_layer.hpp"
#include "factory/layer_factory.hpp"
using namespace TinyTensor;
// int main(){
//   Tensor<float> tensor(3, 32, 32);
//   // ASSERT_EQ(tensor.channels(), 3);
//   // ASSERT_EQ(tensor.rows(), 32);
//   // ASSERT_EQ(tensor.cols(), 32);
//   // Tensor<float> tensor(1,5,5);
//   // //uint32_t a=128;
//   // tensor.Rand();
//   // tensor.Show();
//   // tensor.Flatten();
//   // tensor.Show();
//   // std::vector<uint32_t> b(tensor.shapes());
//   //arma::fcube data(3,12,12);
//   //tensor.set_data(data);
//   // std::cout<<tensor.channels()<<std::endl;
//   // std::cout<<tensor.cols()<<std::endl;
//   // std::cout<<tensor.rows()<<std::endl;
//   // std::cout<<tensor.index(12)<<std::endl;
//   // std::cout<<b.size()<<std::endl;
//   // std::cout<<tensor.empty()<<std::endl;
//   // std::cout<<tensor.size()<<std::endl;
//   //std::cout<<tensor.index(a)<<std::endl;
//   //b = tensor.shapes();
//   // for(auto it:b)  std::cout<<it<<" ";
//   // for(int i=0;i<3;i++)
//   // {
//   //   std::cout<<b[i]<<std::endl;
//   // }
 
//   //putchar(10);
//   return 0;mak
// }

// TEST(test_tensor, create) {
//   //using namespace kuiper_infer;
//   Tensor<float> tensor(3, 32, 32);
//   ASSERT_EQ(tensor.channels(), 3);
//   ASSERT_EQ(tensor.rows(), 32);
//   ASSERT_EQ(tensor.cols(), 32);
// }

// int main(){
//   Tensor<float> tensor(3, 3, 3);
//   std::cout<<tensor.channels();
//   std::cout<<tensor.rows();
//   std::cout<<tensor.cols();

//   tensor.Fill(1.f); // 填充为1
//   tensor.Padding({1, 1, 1, 1}, 0); // 边缘填充为0
//   std::cout<<tensor.rows();
//   std::cout<<tensor.cols();

//   int index = 0;
//   // 检查一下边缘被填充的行、列是否都是0
//   for (int c = 0; c < tensor.channels(); ++c) {
//     for (int r = 0; r < tensor.rows(); ++r) {
//       for (int c_ = 0; c_ < tensor.cols(); ++c_) {
//         if (c_ == 0 || r == 0) {
//           std::cout<<tensor.at(c, r, c_);
//         }
//         index += 1;
//       }
//     }
//   }
//   std::cout<< "Test2 passed!";
//   return 0;
// }

int main(){
  float thresh = 0.f;
  // 初始化一个relu operator 并设置属性
  std::shared_ptr<Operator> sigmod_op = std::make_shared<SigmodOperaor>();
  std::shared_ptr<Layer> sigmod_layer = LayerRegisterer::CreateLayer(sigmod_op);

  // 有三个值的一个tensor<float>
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f; //output对应的应该是0
  input->index(1) = -2.f; //output对应的应该是0
  input->index(2) = 3.f; //output对应的应该是3
  // // 主要第一个算子，经典又简单，我们这里开始！
  // input->index(0) = -1.f;
  // for (int i = 0; i < 1; ++i) {
  //   std::cout<<input->index(0)<<std::endl;
  //   std::cout<<input->index(1)<<std::endl;
  //   std::cout<<input->index(2)<<std::endl;
  // }
  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理

  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  inputs.push_back(input);
  for (int i = 0; i < inputs.size(); ++i) {
    std::cout<<inputs.at(i)->index(0)<<std::endl;
    std::cout<<inputs.at(i)->index(1)<<std::endl;
    std::cout<<inputs.at(i)->index(2)<<std::endl;
  }
  //ReluLayer layer(relu_op);

  //layer.Forwards(inputs, outputs);
  sigmod_layer->Forwards(inputs,outputs);
  std::cout<<outputs.size()<<std::endl; //1
  //std::cout<<input->index(0);
  for (int i = 0; i < outputs.size(); ++i) {
    std::cout<<outputs.at(i)->index(0)<<" "<<(1 / (1 + std::exp(-input->index(0))))<<std::endl;
    

    std::cout<<outputs.at(i)->index(1)<<" "<<(1 / (1 + std::exp(-input->index(1))))<<std::endl;
    //std::cout<<(1 / (1 + std::exp(-input->index(1))));

    std::cout<<outputs.at(i)->index(2)<<" "<<(1 / (1 + std::exp(-input->index(2))))<<std::endl;
  }
  // const std::string &file_path = "../tmp/data2.csv";
  // std::vector<std::string> headers;
  // std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadDataWithHeader(file_path, headers, ',');

  // uint32_t index = 1;
  // uint32_t rows = data->rows();
  // uint32_t cols = data->cols();
  // //LOG(INFO) << "\n" << data;
  // CHECK(rows==3);
  // CHECK(cols==3);
  // CHECK(headers.size()==3);

  // std::cout<<headers.at(0);
  // std::cout<<headers.at(1);
  // std::cout<<headers.at(2);
  // putchar(10);
  // for (uint32_t r = 0; r < rows; ++r) {
  //   for (uint32_t c = 0; c < cols; ++c) {
  //     std::cout<<data->at(0, r, c)<<" ";
  //     index += 1;
  //   }
  //   putchar(10);
  // }
  // Tensor<float> tensor(3, 3, 3);
  // std::cout<< tensor.channels();
  // std::cout<< tensor.rows();
  // std::cout<< tensor.cols();
  // for(auto it:tensor.shapes()){
  //   std::cout<<it<<" ";
  // }

  // std::vector<float> values;
  // for (int i = 0; i < 27; ++i) {
  //   values.push_back((float) i);
  // }
  // tensor.Fill(values);
  // std::cout<<  tensor.data();

  // int index = 0;
  // for (int c = 0; c < tensor.channels(); ++c) {
  //   for (int r = 0; r < tensor.rows(); ++r) {
  //     for (int c_ = 0; c_ < tensor.cols(); ++c_) {
  //       std::cout<< tensor.at(c, r, c_);
  //       index += 1;
  //     }
  //   }
  // }
  // std::cout<< "Test1 passed!";
}