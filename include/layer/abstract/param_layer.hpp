/*
 * @Author: lihaobo
 * @Date: 2023-05-10 01:39:47
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-05-10 01:49:44
 * @Description:
 */

#ifndef TINYTENSOR_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#include "layer.hpp"

namespace TinyTensor
{
    class ParamLayer : public Layer
    {
    public:
        explicit ParamLayer(const std::string &layer_name);

        /**
         * 初始化权重参数
         * @param param_count 参数 B
         * @param param_channel 参数 C
         * @param param_height 参数 H
         * @param param_width 参数 W
        */
        void InitWeightParam(const uint32_t param_count, const uint32_t param_channel, const uint32_t param_height, const uint32_t param_width);

        /**
         * 初始化权重偏置
         * @param param_count 参数 B
         * @param param_channel 参数 C
         * @param param_height 参数 H
         * @param param_width 参数 W
        */
        void InitBiasParam(const uint32_t param_count, const uint32_t param_channel, const uint32_t param_height, const uint32_t param_width);

        /**
         * 重写 Layer::weights ：获取当前层所有权重参数
        */
        const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

        /**
         * 重写 Layer::bias ：获取当前层所有偏置参数
        */
        const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

        /**
         * 重写 Layer::set_weights ：设置当前层所有权重参数
        */
        void set_weights(const std::vector<float> &weights) override;
        void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

        /**
         * 重写 Layer::set_bias ：设置当前层所有偏置参数
        */
        void set_bias(const std::vector<float> &bias) override;
        void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

    protected:
        std::vector<std::shared_ptr<Tensor<float>>> weights_;   // 权重
        std::vector<std::shared_ptr<Tensor<float>>> bias_;      // 偏置
    };

} // namespace TinyTensor

#endif // TINYTENSOR_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_