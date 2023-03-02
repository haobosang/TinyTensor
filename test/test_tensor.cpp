#include "data/Tensor.hpp"
#include<memory>
#include<gtest/gtest.h>
#include<armadillo>
#include<glog/logging.h>
using namespace TinyTensor;
int main(){
  Tensor<float> tensor(3,12,12);
  int a = (1U);
  std::cout<<a<<std::endl;
  std::cout<<tensor.cols()<<std::endl;
  return 0;
}