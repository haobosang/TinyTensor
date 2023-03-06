#ifndef TINYTENSOR_INCLUDE_OPS_OP_HPP_
#define TINYTENSOR_INCLUDE_OPS_OP_HPP_


namespace TinyTensor{
enum class OpType{
    kOperatorUnknown =-1,
    kOperatorRelu = 0,
};

class Operator{
public:
    OpType op_type_ = OpType::kOperatorUnknown;

    virtual ~Operator() = default ;

    explicit Operator(OpType op_type);

};

}

#endif //TINYTENSOR_INCLUDE_OPS_OP_HPP_