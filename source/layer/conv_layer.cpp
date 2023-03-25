/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:49:50
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-25 13:25:53
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

    const auto &weights = this->op_->weight();
    CHECK(!weights.empty());

     

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

    const uint32_t batch_size = inputs.size();
    for(uint32_t i=0;i<batch_size;i++){

        const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty()) << "The input feature map of conv layer is empty";

        std::shared_ptr<Tensor<float>> input_;
        if(padding_h >0||padding_w>0){
            input_ = input->Clone();
            input_->Padding({padding_h, padding_h, padding_w, padding_w},0);
        }else{
            input_ =input;
        }

        const uint32_t input_w = input_->cols();
        const uint32_t input_h = input_->rows();
        const uint32_t input_c = input_->channels();
        const uint32_t kernel_count = weights.size();
        CHECK(kernel_count > 0) << "kernel count must greater than zero";

        uint32_t kernel_h = weights.at(0)->rows();
        uint32_t kernel_w = weights.at(0)->cols();
        CHECK(kernel_h > 0 && kernel_w > 0)
            << "The size of kernel size is less than zero";

        uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
        uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
        CHECK(output_h > 0 && output_w > 0)
            << "The size of the output feature map is less than zero";


    }


    




                        


}

std::shared_ptr<Layer> ConvolutionLayer::CreateInstance(const std::shared_ptr<Operator> &op){
     std::shared_ptr<Layer> conv_layer = std::make_shared<ConvolutionLayer>(op);
    return conv_layer;

}




} // namespace TinyTensor
