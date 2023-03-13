/*
 * @Author: lihaobo
 * @Date: 2023-03-13 14:14:16
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-13 15:24:18
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#define TINYTENSOR_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#include <vector>
#include <memory>
#include <string>
#include <utility>

namespace TinyTensor
{

enum class TokenType{
    TokenUnKnown = -1,
    TokenInputNumber = 0,
    TokenComma = 1,
    TokenAdd = 2,
    TokenMul = 3,
    TokenLeftBracket  = 4,
    TokenRightBracket = 5,
};

struct Token{
    TokenType token_type = TokenType::TokenUnKnown;
    int32_t start_pos = 0;
    int32_t end_pos = 0;
    Token(TokenType token_type, int32_t start_pos, int32_t end_pos):token_type(token_type),start_pos(start_pos),end_pos(end_pos){}
};
struct TokenNode{
    int32_t num_idx = -1;
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;
    TokenNode(int32_t num_idx,std::shared_ptr<TokenNode> left,std::shared_ptr<TokenNode> right);
    ~TokenNode() = default;

};


} // namespace TinyTensor



#endif //TINYTENSOR_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_