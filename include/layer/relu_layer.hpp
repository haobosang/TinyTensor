/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:04:56
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-09 13:45:16
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INCLUDE_RELU_LAYER_HPP_
#define TINYTENSOR_INCLUDE_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"
namespace TinyTensor
{


class ReluLayer : public Layer{
private:

    std::shared_ptr<ReluOperaor> _op;
public:
    explicit ReluLayer(const std::shared_ptr<Operator> &op);
    
    ~ReluLayer() override = default;

    void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
};

    
} // namespace TinyTensor



#endif //TINYTENSOR_INCLUDE_RELU_LAYER_HPP_