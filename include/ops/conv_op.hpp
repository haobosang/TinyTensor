/*
 * @Author: lihaobo
 * @Date: 2023-03-22 19:47:35
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-24 14:32:20
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_OPS_CONV_OP_HPP_
#define TINYTENSOR_OPS_CONV_OP_HPP_
#include "op.hpp"
#include "data/Tensor.hpp"
#include <vector>
namespace TinyTensor
{

    class ConvolutionOp : public Operator
    {

    private:
        bool use_bais_;
        uint32_t groups_ = 1;
        uint32_t padding_h_ = 0;
        uint32_t padding_w_ = 0;
        uint32_t stride_h_ = 1;
        uint32_t stride_w_ = 1;
        std::vector<std::shared_ptr<Tensor<float>>> weight_;
        std::vector<std::shared_ptr<Tensor<float>>> bias_;

    public:
        explicit ConvolutionOp(bool use_bias, uint32_t groups, uint32_t padding_h, uint32_t padding_w,
                               uint32_t stride_h, uint32_t stride_w) : Operator(OpType::kOperatorConvolution), use_bais_(use_bias), groups_(groups),
                                                                       padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w)
        {
        }

        void set_weight(std::vector<std::shared_ptr<Tensor<float>>> &weight);

        void set_baies(std::vector<std::shared_ptr<Tensor<float>>> &bais);

        const std::vector<std::shared_ptr<Tensor<float>>> &weight() const;

        const std::vector<std::shared_ptr<Tensor<float>>> &bais() const;

        bool is_use_bias() const;

        void set_use_bias(bool use_bias);

        uint32_t groups() const;

        void set_groups(uint32_t groups);

        uint32_t padding_h() const;

        void set_padding_h(uint32_t padding_h);

        uint32_t padding_w() const;

        void set_padding_w(uint32_t padding_w);

        uint32_t stride_h() const;

        void set_stride_h(uint32_t stride_h);

        uint32_t stride_w() const;

        void set_stride_w(uint32_t stride_w);
    };

}

#endif // TINYTENSOR_OPS_CONV_OP_HPP_