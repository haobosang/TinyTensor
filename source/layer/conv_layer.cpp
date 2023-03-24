/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:49:50
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 16:37:48
 * @Description: 请填写简介
 */
#include "layer/conv_layer.hpp"
#include "glog/logging.h"

namespace TinyTensor
{

ConvolutionLayer::ConvolutionLayer(const std::shared_ptr<Operator> &op):Layer("Conv"){
    CHECK(op != nullptr && op->op_type_ == OpType::kOperatorConvolution);

    ConvolutionOp *convolution_op = dynamic_cast<ConvolutionOp *>(op.get());

    CHECK(convolution_op!=nullptr)<< "Convolution operator is empty"; 
    this->op_ = std::make_shared<ConvolutionOp>(*convolution_op);


}

void ConvolutionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs){
    //CHECK(!inputs.size());
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorConvolution);


                        


}

std::shared_ptr<Layer> ConvolutionLayer::CreateInstance(const std::shared_ptr<Operator> &op){
     std::shared_ptr<Layer> conv_layer = std::make_shared<ConvolutionLayer>(op);
    return conv_layer;

}




} // namespace TinyTensor
