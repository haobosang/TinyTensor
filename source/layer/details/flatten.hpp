#ifndef TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/layer.hpp"

namespace TinyTensor
{
    class FlattenLayer : public Layer
    {
    public:
        explicit FlattenLayer(int start_dim, int end_dim);

        InferStatus Forward
        (
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs
        ) override;

        static ParseParameterAttrStatus GetInstance
        (
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &flatten_layer
        );

    private:
        int start_dim_ = 0;
        int end_dim_ = 0;
    };
}
#endif // TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
