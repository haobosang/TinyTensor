#ifndef TINYTENSOR_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/layer.hpp"

namespace TinyTensor
{
    class SoftmaxLayer : public Layer
    {
    public:
        explicit SoftmaxLayer();

        InferStatus Forward
        (
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs
        ) override;
    };
}

#endif // TINYTENSOR_INFER_SOURCE_LAYER_SOFTMAX_HPP_
