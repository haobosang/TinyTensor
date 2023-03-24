#include "ops/conv_op.hpp"

namespace TinyTensor{

void ConvolutionOp::set_weight(std::vector<std::shared_ptr<Tensor<float>>> &weight){
    this->weight_ = weight;

}
void ConvolutionOp::set_baies(std::vector<std::shared_ptr<Tensor<float>>> &bais){
    this->bias_ = bais;
}

const std::vector<std::shared_ptr<Tensor<float>>> &ConvolutionOp::weight() const{
    return this->weight_;

}
const std::vector<std::shared_ptr<Tensor<float>>> &ConvolutionOp::bais() const{
    return this->bias_;

}
bool ConvolutionOp::is_use_bias() const{
    return this->use_bais_;
}
void ConvolutionOp::set_use_bias(bool use_bias){
    this->use_bais_ = use_bias;
}
uint32_t ConvolutionOp::groups() const{
    return this->groups_;
}
void ConvolutionOp::set_groups(uint32_t groups){
    this->groups_ = groups;

}
uint32_t ConvolutionOp::padding_h() const{
    return this->padding_h_;

}
void ConvolutionOp::set_padding_h(uint32_t padding_h){
    this->padding_h_ = padding_h;

}
uint32_t ConvolutionOp::padding_w() const{
    return this->padding_w_;

}

void ConvolutionOp::set_padding_w(uint32_t padding_w){
    this->padding_w_ = padding_w;

}

uint32_t ConvolutionOp::stride_h() const{
    return this->stride_h_;

}

void ConvolutionOp::set_stride_h(uint32_t stride_h){
    this->stride_h_ = stride_h;
}

uint32_t ConvolutionOp::stride_w() const{
    return this->stride_w_;
}

void ConvolutionOp::set_stride_w(uint32_t stride_w){
    this->stride_w_ = stride_w;
}

}