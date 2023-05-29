#ifndef TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/layer.hpp"

namespace TinyTensor
{

class FlattenLayer : public Layer
{
public:
    /**
     * 构造函数
     * @param start_dim 开始维度
     * @param end_dim 结束维度
     */
    explicit FlattenLayer(int32_t start_dim, int32_t end_dim);

    /**
     * 前馈成员函数
     * @param inputs 输入一个 Tensor
     * @param outputs 输出的 Tensor
     */
    InferStatus Forward
    (
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs
    ) override;

    /**
     * 获取实例
     * @param op 计算图中的计算节点
     * @param flatten_layer 输入 flatten_layer 层
     */
    static ParseParameterAttrStatus GetInstance
    (
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &flatten_layer
    );

private:
    int32_t start_dim_ = 0;     // 开始维度
    int32_t end_dim_ = 0;       // 结束维度
};

}
#endif // TINYTENSOR_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
