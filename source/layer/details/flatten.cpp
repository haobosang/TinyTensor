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
    if (inputs.empty()) // 1. 输入不能为空
    {
        LOG(ERROR) << "The input feature map of flatten layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) // 2. 要求输入输出尺度相同
    {
        LOG(ERROR) << "The input and output size of flatten layer is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    uint32_t start_dim = this->start_dim_;  // 获取开始维度
    uint32_t end_dim = this->end_dim_;      // 获取结束维度
    const uint32_t total_dims = 4;          // 总维度为 4

    if (start_dim < 0) // 开始维度为负数的情况，从后往前数
    {
        start_dim = total_dims + start_dim; 
    }
    if (end_dim < 0)   // 结束维度为负数的情况，从后往前数
    {
        end_dim = total_dims + end_dim;
    }

    // 检查开始维度和结束维度的合法性
    CHECK(end_dim > start_dim) << "The end dim must greater than start dim";
    CHECK(end_dim <= 3 && start_dim >= 1) << "The end dim must less than two and start dim must greater than zero";
    
    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) // batch 操作
    {
        const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
        if (input == nullptr || input->empty())
        {
            LOG(ERROR) << "The input tensor array in the flatten layer has an empty tensor" << i << " th";
            return InferStatus::kInferFailedInputEmpty;
        }

        std::vector<uint32_t> shapes = input->shapes(); 
        shapes.insert(shapes.begin(), batch_size); // 扩充为 4 维

        // 计算从开始维度到结束维度的元素个数
        uint32_t elem_size = std::accumulate(shapes.begin() + start_dim, shapes.begin() + end_dim + 1, 1, std::multiplies<uint32_t>());
        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        output = input->Clone();

        CHECK(input->size() == output->size()) << "The output and input shapes of the flatten layer do not match" << i << " th";
        outputs.at(i) = output;

        if (start_dim == 1 && end_dim == 3)
        {
            output->ReRawshape({elem_size});
        }
        else if (start_dim == 2 && end_dim == 3)
        {
            uint32_t channels = input->channels();
            output->ReRawshape({channels, elem_size});
        }
        else if (start_dim == 1 && end_dim == 2) 
        {
            uint32_t cols = input->cols();
            output->ReRawshape({elem_size, cols});
        }
        else 
        {
            LOG(FATAL) << "Wrong faltten dim: \nstart_dim: " << start_dim << "end_dim: " << end_dim;
        }
    }
    return InferStatus::kInferSuccess;
}

/* 获取运行时的计算图实例 */
ParseParameterAttrStatus FlattenLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &flatten_layer)
{
    CHECK(op != nullptr) << "Flatten operator is nullptr";

    const std::map<std::string, TinyTensor::RuntimeParameter*>& params = op->params;
    if(params.find("end_dim") == params.end()) 
    {
        LOG(ERROR) << "Cat not find the dimension parameter";
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    if (params.find("start_dim") == params.end())
    {
        LOG(ERROR) << "Can not find the dimension parameter";
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    const auto& start_dim = dynamic_cast<RuntimeParameterInt*>(params.at("start_dim"));
    const auto& end_dim = dynamic_cast<RuntimeParameterInt*>(params.at("end_dim"));

    if (start_dim == nullptr || end_dim == nullptr)
    {
        return ParseParameterAttrStatus::kParameterMissingDim;
    }

    flatten_layer = std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper FlattenGetInstance("torch.flatten", FlattenLayer::GetInstance);

} // namespace TinyTensor