/**
 * LinearLayer 类的实现
 * @author PMSang
 * @date 20230529
*/

#include "linear.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace TinyTensor 
{

/* 构造函数，使用初始化参数列表构建 Linear 层*/
LinearLayer::LinearLayer(uint32_t inplane, uint32_t plane, bool bias) : ParamLayer("Linear"), in_features_(inplane), out_features_(plane), use_bias_(bias) 
{
    // 首先初始化权重参数
    this->InitWeightParam(1, 1, plane, inplane);
    if (bias)
    {
        this->InitBiasParam(1, 1, 1, plane);
    }
}

/* 前馈成员函数 */
InferStatus LinearLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs)
{
    if (inputs.empty()) // linear 层的输入张量为空
    {
        LOG(ERROR) << "The input tensor array in the linear layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) // 输入输出张量尺度不匹配
    {
        LOG(ERROR) << "The input and output tensor array size of linear layer do not match";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    if (this->weights_.empty()) // 权重未初始化
    {
        LOG(ERROR) << "The weight tensor in the linear layer is empty";
        return InferStatus::kInferFailedWeightParameterError;
    } 
    else 
    {
        if (this->use_bias_ && this->weights_.size != this->bias_.size()) // 偏置与权重尺寸不匹配
        {
            LOG(ERROR) << "The size of the weight and bias tensor do not match";
            return InferStatus::kInferFailedBiasParameterError;
        }
    }

    if (this->weights_.size() != 1) // 权重的 batch_size 不为 1
    {
        LOG(ERROR) << "Need one weight tensor in the linear layer";
        return InferStatus::kInferFailedWeightParameterError;
    }

    if (this->use_bias_ && this->bias_.size() != 1) // 偏置 batch_size 不为 1
    {
        LOG(ERROR) << "Need one bias tensor in the linear layer";
        return InferStatus::kInferFailedBiasParameterError;
    }

    uint32_t batch_size inputs.size();
    const std::shared_ptr<Tensor<float>>& weight = this->weights_.front(); // 第一个 batch 的权重

    /* ?? */
    arma::fmat weight_data(weight->raw_ptr(), this->out_features_, this->in_features_, false, true);
    const arma::fmat& weigth_data_t = weight_data.t();

#pragma omp parallel for num_threads(batch_size) // batch 并行
    for (uint32_t i = 0; i < batch_size; ++i)
    {
        // 获得当前 batch 的 fcube 张量
        const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
        CHECK(input != nullptr && !input->empty()) << "The input tensor array in the linear layer has an empty tensor " << i << "-th";
        const std::vector<uint32_t>& input_shapes = input->shapes(); // [1, c, h, w]

        const uint32_t feature_dims = input_shapes.at(1);   // c
        const uint32_t in_features = input_shapes.at(2);    // h
        CHECK(weight_data.n_rows == this->out_features_) << "The row of weight tensor should be same to output_feature_";
        CHECK(weight_data.n_cols == in_features && in_features == this->in_features_) << "The col of weoght tensor should be same to input_feature_";

        /* 根据 input 数据，按列创建矩阵对象 */
        arma::fmat input_vec((float*)input->raw_ptr(), feature_dims, this->in_features_, false, true);

        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) // 输出不能为空
        {
            output = std::make_shared<Tensor<float>>(1, this->out_features_, feature_dims);
            outputs.at(i) = output;
        }
        CHECK(output->channels() == 1 && output->rows() == feature_dims && output->cols() == this->out_features_)
            << "The row of output tessor should be same to feature_dims_ and the col of output tensor should be same to output_feature_ " << i << "-th";

        const auto& output_raw_shapes = output->shapes(); // 形状
        if (output_raw_shapes.size() == 2) 
        { // 两个维度的张量
            CHECK(output_raw_shapes.at(0) == feature_dims && output_raw_shapes.at(1) == this->out_features_);
        }
        if (output_raw_shapes.size() == 1) 
        { // 一个维度的张量
            CHECK(output_raw_shapes.at(0) == out_features_);
        }

        arma::fmat& result = output->slice(0); // 拷贝一个第 0 维度的 tensor 作为结果
        result = input_vec * weight_data_t; // 输入 * 权重 = 输出结果

        if (use_bias_) // 使用偏置
        {
            CHECK(!this->bias_.empty() && this->bias_.size() == 1) << "The bias tensor is empty, but use_bias is true";

            const auto& bias_data = this->bias_.front()->data();
            CHECK(!bias_data.empty() && bias_data.n_slices == 1 && bias_data.n_cols == this->out_features_) << "The col of bias tensor is not same to output_features_";
            const auto& bias_tensor = bias_data.slice(0);

#pragma omp parallel for
            for (uint32_t row = 0; row < result.n_rows; ++row) 
            {
                result.row(row) += bias_tensor; // 乘完的结果 + 偏置
            }
        }
    }
    return InferStatus::kInferSuccess;
}

/* 获取计算图实例 */
ParseParameterAttrStatus LinearLayer::GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& linear_layer) 
{
    CHECK(op != nullptr) << "Linear operator is nullptr";
    const auto& params = op->params; // 操作算子的参数
    if (params.find("bias") == params.end()) // 没找到 bias 参数
    {
        LOG(ERROR) << "Can not find the use bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    } 

    const auto& use_bias_param = dynamic_cast<RuntimeParameterBool*>(params.at("bias")); // 运行时转换 bias 类型：bool* -> RuntimeParameterBool*
    if (use_bias_param == nullptr) // 没找到 bias 参数
    {
        LOG(ERROR) << "Can not find the use bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    }

    const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attr = op->attribute; // 获取当前操作算子的所有属性
    CHECK(!attr.empty()) << "Operator attributes is empty";

    if (attr.find("weight") == attr.end())  // 找不到权重参数
    {
        LOG(ERROR) << "Can not find the weight parameter";
        return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    if (use_bias_param->value) // 偏置参数有值
    {
        if (attr.find("bias") == attr.end()) // 但计算图的注册表中没发现 bias 层
        {
            LOG(ERROR) << "Can not find the bias parameter";
            return ParseParameterAttrStatus::kAttrMissingBias;
        }
    }

    const auto& weight = attr.at("weight");
    const auto& bias = attr.at("bias");
    const auto& shapes = weight->shape;
    if ((shapes.size() < 2))  // 权重矩阵的size不应该小于两维度
    {
        LOG(ERROR) << "The graph only support two dimension matrix multiply";
        return ParseParameterAttrStatus::kAttrMissingOutFeatures;
    }

    /* 初始化 linear 层 */

    int32_t out_features = shapes.at(0);
    int32_t in_features = shapes.at(1);
    const bool use_bias = use_bias_param->value;

    linear_layer = std::make_shared<LinearLayer>(in_features, out_features, use_bias);
    if (use_bias) 
    {
        linear_layer->set_bias(bias->get<float>());
    }
    
    /* 加载权重 */

    linear_layer->set_weights(weight->get<float>());
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

/* 注册计算图到注册表 */
LayerRegistererWrapper kLinearGetInstance("nn.Linear", LinearLayer::GetInstance);

}