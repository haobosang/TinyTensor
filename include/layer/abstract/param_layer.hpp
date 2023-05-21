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

        void InitWeightParam(const uint32_t param_count, const uint32_t param_channel, const uint32_t param_height, const uint32_t param_width);

        void InitBiasParam(const uint32_t param_count, const uint32_t param_channel, const uint32_t param_height, const uint32_t param_width);

        const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

        const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

        void set_weights(const std::vector<float> &weights) override;

        void set_bias(const std::vector<float> &bias) override;

        void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

        void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

    protected:
        std::vector<std::shared_ptr<Tensor<float>>> weights_;
        std::vector<std::shared_ptr<Tensor<float>>> bias_;
    };

} // namespace TinyTensor

#endif // TINYTENSOR_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_