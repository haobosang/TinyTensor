/*
 * @Author: lihaobo
 * @Date: 2023-03-16 15:27:56
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-20 11:02:17
 * @Description: 请填写简介
 */
#include "ops/expression_op.hpp"
#include "glog/logging.h"
namespace TinyTensor
{
ExpressionOp::ExpressionOp(const std::string &expr):Operator(OpType::kExpression),expr_(expr){
    this->parser_ = std::make_shared<ExpressionParser>(this->expr_);
}
std::vector<std::shared_ptr<TokenNode>> ExpressionOp::Generate(){
    CHECK(this->parser_ !=nullptr);
    this->nodes_ = this->parser_->Generate();
    return this->nodes_;


}
} // namespace TinyTensor
