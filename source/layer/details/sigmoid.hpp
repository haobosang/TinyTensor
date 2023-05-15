/*
 * @Author: panmeng
 * @Date: 2023-05-15 13:29:14
 * @LastEditors: panmeng
 * @LastEditTime: 2023-05-15 13:29:14
 * @Description: 请填写简介
 */

#ifndef TINYTENSOR_INCLUDE_SIGMOID_LAYER_HPP_
#define TINYTENSOR_INCLUDE_SIGMOID_LAYER_HPP_

#include "layer/abstract/layer.hpp"

namespace TinyTensor
{
/**
 * 公共继承自 Layer 虚基类
*/
class SigmoidLayer: public Layer
{
public:
    SigmoidLayer(): Layer("Sigmoid") { }
    ~SigmoidLayer() = default;

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &sigmoid_layer);
};

} // namespace TinyTensor

#endif //TINYTENSOR_INCLUDE_SIGMOID_LAYER_HPP_