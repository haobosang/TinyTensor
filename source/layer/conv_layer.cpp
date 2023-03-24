/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:49:50
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 20:54:00
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
    
    CHECK(!inputs.size())<< "Input is empty!";;
    CHECK(inputs.size()!=outputs.size());

    const auto &weight = this->op_->weight();
    CHECK(!weight.empty());

     

    std::vector<std::shared_ptr<Tensor<float> >> bias_;
    if (this->op_->is_use_bias()) {
    bias_ = this->op_->bais();
    }
    
    const uint32_t stride_h = this->op_->stride_h();
    const uint32_t stride_w = this->op_->stride_w();
    CHECK(stride_w > 0 && stride_h > 0);
    const uint32_t padding_h = this->op_->padding_h();
    const uint32_t padding_w = this->op_->padding_w();
    const uint32_t groups = this->op_->groups();

    




                        


}

std::shared_ptr<Layer> ConvolutionLayer::CreateInstance(const std::shared_ptr<Operator> &op){
     std::shared_ptr<Layer> conv_layer = std::make_shared<ConvolutionLayer>(op);
    return conv_layer;

}




} // namespace TinyTensor
