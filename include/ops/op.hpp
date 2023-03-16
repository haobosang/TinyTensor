/*
 * @Author: haobosang lihaobo1998@gmail.com
 * @Date: 2023-03-06 10:05:14
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-16 15:55:34
 * @FilePath: /MyTinyTensor/include/ops/op.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef TINYTENSOR_INCLUDE_OPS_OP_HPP_
#define TINYTENSOR_INCLUDE_OPS_OP_HPP_


namespace TinyTensor{
enum class OpType{
    kOperatorUnknown =-1,
    kOperatorRelu = 0,
    kOperatorSigmod = 1,
    kExpression = 2,
    
};

class Operator{
public:
    OpType op_type_ = OpType::kOperatorUnknown;

    virtual ~Operator() = default ;

    explicit Operator(OpType op_type);

};

}

#endif //TINYTENSOR_INCLUDE_OPS_OP_HPP_