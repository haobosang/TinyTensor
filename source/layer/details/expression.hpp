/*
 * @Author: lihaobo
 * @Date: 2023-03-16 15:30:46
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 15:22:27
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_
#define TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_

#include "layer/abstract/layer.hpp"
#include "parser/parse_expression.hpp"

namespace TinyTensor
{
    class ExpressionLayer : public Layer
    {
    public:
        explicit ExpressionLayer(const std::string &statement);

        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs
        ) override;

        static ParseParameterAttrStatus GetInstance
        (
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &expression_layer
        );

    private:
        std::unique_ptr<ExpressionParser> parser_;
    };
}

#endif // TINYTENSOR_INFER_INCLUDE_LAYER_EXPRESSION_LAYER_HPP_