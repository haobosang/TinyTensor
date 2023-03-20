/*
 * @Author: lihaobo
 * @Date: 2023-03-16 15:44:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-20 12:51:12
 * @Description: 请填写简介
 */

#include "layer/expression_layer.hpp"
#include "glog/logging.h"
#include "ops/op.hpp"
#include <stack>
namespace TinyTensor
{
ExpressionLayer::ExpressionLayer(const std::shared_ptr<Operator> &op):Layer("Expression"){
    CHECK( op!=nullptr && op->op_type_ == OpType::kExpression);
    ExpressionOp *expression_op = dynamic_cast<ExpressionOp *>(op.get());

    CHECK(expression_op!=nullptr)<< "Expression operator is empty"; 
    this->op_ = std::make_shared<ExpressionOp>(*expression_op);


}
void ExpressionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &input,std::vector<std::shared_ptr<Tensor<float>>> &output){
    CHECK(!input.empty());

    const uint32_t batch_size = output.size();

    CHECK(batch_size !=0 );

    for(int i=0;i<batch_size;i++)
    {
        CHECK(output.at(i)!=nullptr && !output.at(i)->empty());
        output.at(i)->Fill(0.f);
    }
    CHECK(this->op_!=nullptr && this->op_->op_type_ == OpType::kExpression);
    std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
    const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->op_->Generate();

    for(const auto &token_node:token_nodes){
        if(token_node->num_idx>=0)
        {
            uint32_t start_pos = token_node->num_idx*batch_size;
            std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
            for(uint i=0;i<batch_size;i++)
            {
                CHECK(i+start_pos<input.size());
                input_token_nodes.push_back(input.at(i+start_pos));
            }
            op_stack.push(input_token_nodes);
        }else{
            const int32_t op = token_node->num_idx;
            CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
            std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();
            CHECK(input_node1.size() == batch_size);
            
            op_stack.pop();

            std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
            CHECK(input_node2.size() == batch_size);
            op_stack.pop();

            CHECK(input_node1.size()== input_node2.size());
            std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
            for(uint32_t i =0;i<batch_size;i++)
            {
                if(op == -int(TokenType::TokenAdd)){
                    output_token_nodes.at(i) = Tensor<float>::ElementAdd(input_node1.at(i),input_node2.at(i));
                }else if(op == -int(TokenType::TokenMul)){
                     output_token_nodes.at(i) = Tensor<float>::ElementMul(input_node1.at(i),input_node2.at(i));
                }else{
                    LOG(FATAL) << "Unknown operator type: " << op;
                }
            }
            op_stack.push(output_token_nodes);


        }
        
    }
    CHECK(op_stack.size() == 1);
    std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
    op_stack.pop();
    for (int i = 0; i < batch_size; ++i) {
        CHECK(output.at(i) != nullptr && !output.at(i)->empty());
        output.at(i) = output_node.at(i);
    }

}

} // namespace TinyTensor


