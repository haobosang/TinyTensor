/**
 * @Author: PanMeng
 * @Date: 2023-05-10 01:39:47
 * @LastEditors: PMSang
 * @LastEditTime: 2023-05-21 03:03:20
 * @Description:
 */

#include "flatten.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include <functional>
namespace TinyTensor 
{

/* 构造函数实现 (使用初始化参数列表去实现构造) */
FlattenLayer::FlattenLayer(int32_t start_dim, int32_t end_dim) : Layer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) 
{
    // Layer("Flatten");
    // this->start_dim_ = start_dim;
    // this->end_dim_ = end_dim;
}

/* 前馈成员函数实现 */
InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs)
{
    if (inputs.empty()) 
    {
        LOG(ERROR) << "The input tensor array in the flatten layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) 
    {
        LOG(ERROR) << "The input and output tensor array size of the flatten layer do not match";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    int start_dim = start_dim_;
    int end_dim = end_dim_;
    int total_dims = 4;  // NCHW

    if (start_dim < 0) 
    {
        start_dim = total_dims + start_dim;
    }
    if (end_dim < 0) 
    {
        end_dim = total_dims + end_dim;
    }

    CHECK(end_dim > start_dim) << "The end dim must greater than start dim";
    CHECK(end_dim <= 3 && start_dim >= 1) << "The end dim must less than two and start dim must greater than zero";

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) 
    {
        const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
        if (input == nullptr || input->empty()) 
        {
            LOG(ERROR) << "The input tensor array in the flatten layer has an empty tensor " << i << " th";
            return InferStatus::kInferFailedInputEmpty;
        }

        auto shapes = input->shapes();
        shapes.insert(shapes.begin(), batch_size);
        uint32_t elements_size = std::accumulate(shapes.begin() + start_dim, shapes.begin() + end_dim + 1, 1, std::multiplies());

        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        output = TensorClone(input);
        CHECK(input->size() == output->size()) << "The output and input shapes of the flatten layer do not match " << i << " th";
        outputs.at(i) = output;

        if (start_dim == 1 && end_dim == 3) 
        {
            output->Reshape({elements_size}, true);
        } 
        else if (start_dim == 2 && end_dim == 3) 
        {
            uint32_t channels = input->channels();
            output->Reshape({channels, elements_size}, true);
        } 
        else if (start_dim == 1 && end_dim == 2) 
        {
            uint32_t cols = input->cols();
            output->Reshape({elements_size, cols}, true);
        } 
        else 
        {
            LOG(FATAL) << "Wrong flatten dim: " << "start dim: " << start_dim << " end dim: " << end_dim;
        }
    }
    return InferStatus::kInferSuccess;
}

/* 获取运行时的计算图实例 */
ParseParameterAttrStatus FlattenLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &flatten_layer)
{
    CHECK(op != nullptr) << "Flatten operator is nullptr";
    const auto &params = op->params;

    if (params.find("end_dim") == params.end()) 
    {
        LOG(ERROR) << "Can not find the dimension parameter";
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    if (params.find("start_dim") == params.end()) 
    {
        LOG(ERROR) << "Can not find the dimension parameter";
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    const auto &start_dim = dynamic_cast<RuntimeParameterInt *>(params.at("start_dim"));

    const auto &end_dim = dynamic_cast<RuntimeParameterInt *>(params.at("end_dim"));

    if (start_dim == nullptr || end_dim == nullptr) 
    {
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    flatten_layer = std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper FlattenGetInstance("torch.flatten", FlattenLayer::GetInstance);

} // namespace TinyTensor