/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:49:26
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 15:22:05
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_LAYER_CONV_LAYER_HPP_
#define TINYTENSOR_INFER_INCLUDE_LAYER_CONV_LAYER_HPP_

#include "layer.hpp"
#include "ops/conv_op.hpp"
namespace TinyTensor
{

    class ConvolutionLayer : public Layer
    {
    private:
        /* data */
        std::shared_ptr<ConvolutionOp> op_;

    public:
        explicit ConvolutionLayer(const std::shared_ptr<Operator> &op);

        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

        //~ConvolutionLayer();
    };

}

#endif // TINYTENSOR_INFER_INCLUDE_LAYER_CONV_LAYER_HPP_