/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:08:22
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-06 17:20:46
 * @Description: 请填写简介
 */
#include "ops/relu_op.hpp"

namespace TinyTensor{

ReluOperaor::ReluOperaor(float thresh):thresh_(thresh),Operator(OpType::kOperatorRelu){ }

void ReluOperaor::set_thresh(float thresh){
    this->thresh_ = thresh;
    return ;
}

float ReluOperaor::get_thresh() const{
    return this->thresh_;
}

}
