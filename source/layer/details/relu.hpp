/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:04:56
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-05-11 02:58:19
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INCLUDE_RELU_LAYER_HPP_
#define TINYTENSOR_INCLUDE_RELU_LAYER_HPP_

#include "layer/abstract/layer.hpp"
namespace TinyTensor
{

class ReluLayer : public Layer
{
public:
    ReluLayer() : Layer("Relu") { }

    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs
    ) override;

    static ParseParameterAttrStatus GetInstance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &relu_layer
    );
};

} // namespace TinyTensor

#endif // TINYTENSOR_INCLUDE_RELU_LAYER_HPP_