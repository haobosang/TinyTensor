#ifndef TINYTENSOR_INCLUDE_MAXPOOLING_LAYER_HPP_
#define TINYTENSOR_INCLUDE_MAXPOOLING_LAYER_HPP_

#include "layer/abstract/layer.hpp"
namespace TinyTensor{

    class MaxPoolingLayer : public Layer
    {
    private:
        uint32_t pooling_h_ = 0; // 池化核高度大小
        uint32_t pooling_w_ = 0; // 池化核宽度大小
        uint32_t stride_h_ = 1;  // 高度上的步长
        uint32_t stride_w_ = 1;  // 宽度上的步长
        uint32_t padding_h_ = 0; // 高度上的填充
        uint32_t padding_w_ = 0; // 宽度上的填充
        /* data */
    public:
        explicit MaxPoolingLayer
        (
            uint32_t padding_h, 
            uint32_t padding_w, 
            uint32_t pooling_size_h,
            uint32_t pooling_size_w, 
            uint32_t stride_h, 
            uint32_t stride_w
        );

        InferStatus Forward
        (
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs
        ) override;

        static ParseParameterAttrStatus GetInstance
        (
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &max_layer
        );
    };

}

#endif //TINYTENSOR_INCLUDE_MAXPOOLING_LAYER_HPP_