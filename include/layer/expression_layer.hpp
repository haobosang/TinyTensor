/*
 * @Author: lihaobo
 * @Date: 2023-03-16 15:30:46
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-16 16:14:59
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_
#define TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_

#include "layer.hpp"
#include "ops/expression_op.hpp"
#include "data/Tensor.hpp"
namespace TinyTensor
{
class ExpressionLayer :public Layer
{
private:
    /* data */
    std::shared_ptr<ExpressionOp> op_;
public:
    explicit ExpressionLayer(const std::shared_ptr<Operator> &op);
    void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &input,std::vector<std::shared_ptr<Tensor<float>>> &output) override;

};




} // namespace TinyTensor



#endif //TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_