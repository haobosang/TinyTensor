/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:47:35
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-23 15:28:34
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_OPS_CONV_OP_HPP_
#define TINYTENSOR_OPS_CONV_OP_HPP_
#include "op.hpp"
#include "data/Tensor.hpp"
#include <vector>
namespace TinyTensor{

class ConvolutionOp : public Operator{

private:
    bool bais;
    uint32_t groups_ = 1;
    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;
    uint32_t stride_h_ = 0;
    uint32_t stride_w_ = 0;
    std::vector<Tensor<float>> weight_;
    std::vector<Tensor<float>> bias_;
public:
    explicit ConvolutionOp():Operator(OpType::kOperatorConvolution),{}





};

}


#endif //TINYTENSOR_OPS_CONV_OP_HPP_