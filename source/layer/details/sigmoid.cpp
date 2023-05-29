/*
 * @Author: panmeng
 * @Date: 2023-05-15 13:29:14
 * @LastEditors: panmeng
 * @LastEditTime: 2023-05-15 13:29:14
 * @Description: 请填写简介
 */

#include "sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace TinyTensor
{
    
InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs)
{
    if (inputs.empty()) 
    {
        LOG(ERROR) << "The input feature map of sigmoid layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }
    if (inputs.size() != outputs.size()) 
    {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    /* batch 检查*/
    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) 
    {
        const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
        const std::shared_ptr<Tensor<float>> &output_data = outputs.at(i);
        /* 判断是否为空 */
        if (input_data == nullptr || input_data->empty()) 
        {
            LOG(ERROR) << "The input feature map of sigmoid layer is empty";
            return InferStatus::kInferFailedInputEmpty;
        }
        /* 判断 outputs 和 inputs 的大小是否相等*/
        if (output_data != nullptr && !output_data->empty()) {
            if (output_data->shapes() != input_data->shapes()) {
                LOG(ERROR) << "The input and output size is not adapting";
                return InferStatus::kInferFailedInputOutSizeAdaptingError;
            }
        }
    }

/* OMP 并行 */
#pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>> &input_batch = inputs.at(i);
        CHECK(input_batch == nullptr || !input_batch->empty()) << "The input feature map of sigmoid layer is empty";

        std::shared_ptr<Tensor<float>> output_batch = outputs.at(i);
        if (output_batch == nullptr || output_batch->empty())
        {
            DLOG(ERROR) << "The output size of sigmoid is error";
            output_batch = std::make_shared<Tensor<float>>(input_batch->shapes());
            outputs.at(i) = output_batch;
        }
        CHECK(output_batch->shapes() == input_batch->shapes()) << "The output size of sigmoid is error";
        output_batch->set_data(input_batch->data());
        output_batch->data().transform(
            [&](float value) {
                return 1 / (1 + std::exp(-value));
            }
        );  // Tensor.data() 返回的是 arma::fcube 类型的数据. transform 数据变换需要传入变换函数
        // for (float& value : output_batch->data()) {
        //     value = 1 / (1 + exp(-value));
        // }
    }

    return InferStatus::kInferSuccess;
}
    
ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &sigmoid_layer)
{
    CHECK(op != nullptr) << "Sigmoid operator is nullptr";
    sigmoid_layer = std::make_shared<SigmoidLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);

} // namespace TinyTensor
