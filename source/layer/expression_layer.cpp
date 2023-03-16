/*
 * @Author: lihaobo
 * @Date: 2023-03-16 15:44:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-16 16:06:59
 * @Description: 请填写简介
 */

#include "layer/expression_layer.hpp"
#include "glog/logging.h"
#include "ops/op.hpp"

namespace TinyTensor
{
ExpressionLayer::ExpressionLayer(const std::shared_ptr<Operator> &op):Layer("Expression"){
    CHECK( op!=nullptr && op->op_type_ == OpType::kExpression);
    ExpressionOp *expression_op = dynamic_cast<ExpressionOp *>(op.get());

    CHECK(expression_op!=nullptr)<< "Expression operator is empty";
    this->op_ = std::make_shared<ExpressionOp>(*expression_op);


}
void ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,std::vector<std::shared_ptr<Tensor<float>>> &output){

}

} // namespace TinyTensor


