/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:49:50
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 15:37:18
 * @Description: 请填写简介
 */
#include "layer/conv_layer.hpp"
#include "glog/logging.h"

namespace TinyTensor
{

ConvolutionLayer::ConvolutionLayer(const std::shared_ptr<Operator> &op):Layer("Conv"){
    CHECK(op != nullptr && op->op_type_ == OpType::kOperatorConvolution);

}

void ConvolutionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs){
                        


}

std::shared_ptr<Layer> ConvolutionLayer::CreateInstance(const std::shared_ptr<Operator> &op){


    
}




} // namespace TinyTensor
