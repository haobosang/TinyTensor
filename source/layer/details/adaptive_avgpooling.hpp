#ifndef TINYTENSOR_INCLUDE_AVGPOOLING_LAYER_HPP_
#define TINYTENSOR_INCLUDE_AVGPOOLING_LAYER_HPP_

#include "layer/abstract/layer.hpp"

namespace TinyTensor{

    class AdaptiveAveragePoolingLayer : public Layer
    {
    private:
        /* data */
        uint32_t output_h_ = 0;
        uint32_t output_w_ = 0;
    public:
        explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

        InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &avg_layer);
    };

}


#endif //TINYTENSOR_INCLUDE_AVGPOOLING_LAYER_HPP_