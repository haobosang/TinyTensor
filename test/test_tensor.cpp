#include "data/Tensor.hpp"
#include<memory>
#include<gtest/gtest.h>
#include<armadillo>
#include<glog/logging.h>
using namespace TinyTensor;
int main(){
  Tensor<float> tensor(3,12,12);
  //arma::fcube data(3,12,12);
  //tensor.set_data(data);
  std::cout<<tensor.channels()<<std::endl;
  std::cout<<tensor.cols()<<std::endl;
  std::cout<<tensor.rows()<<std::endl;
  
  std::cout<<tensor.empty()<<std::endl;
  std::cout<<tensor.size()<<std::endl;
  return 0;
}