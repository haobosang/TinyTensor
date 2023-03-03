#include "data/Tensor.hpp"
#include<memory>
#include<gtest/gtest.h>
#include<armadillo>
#include<glog/logging.h>
#include<vector>
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

  Tensor<float> tensor(3, 3, 3);
  std::cout<< tensor.channels();
  std::cout<< tensor.rows();
  std::cout<< tensor.cols();

  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back((float) i);
  }
  tensor.Fill(values);
  std::cout<<  tensor.data();

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        std::cout<< tensor.at(c, r, c_);
        index += 1;
      }
    }
  }
  std::cout<< "Test1 passed!";
}