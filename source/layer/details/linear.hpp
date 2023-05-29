#ifndef TINYTENSOR_INFER_SOURCE_LAYER_LINEAR_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace TinyTensor
{

class LinearLayer : public ParamLayer
{
public:
    /**
     * 构造函数
     * @param in_feature 输入特征数
     * @param out_feature 输出特征数
     * @param use_bias 使用偏置
    */
    explicit LinearLayer(uint32_t in_features, uint32_t out_features, bool use_bias);

    /**
     * 重写基类 ParamLayer 前馈成员函数
     * @param inputs 输入 Tensor
     * @param outputs 输出 Tensor
    */
    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    /**
     * 静态成员函数：获取实列
     * @param op 计算图中的计算节点
     * @param flatten_layer 输入 linear_layer 层
    */
    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &linear_layer);

private:
    uint32_t in_features_ = 0;
    uint32_t out_features_ = 0;
    bool use_bias_ = false;
};  

}

#endif // TINYTENSOR_INFER_SOURCE_LAYER_LINEAR_HPP_
