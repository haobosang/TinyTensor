#include "ops/relu_op.hpp"

namespace TinyTensor{

ReluOperaor::ReluOperaor(float thresh):thresh_(thresh){ }

void ReluOperaor::set_thresh(float thresh){
    this->thresh_ = thresh;
    return ;
}

float ReluOperaor::get_thresh() const{
    return this->thresh_;
}

}
