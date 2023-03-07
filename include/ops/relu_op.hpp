#ifndef TINYTENSOR_INCLUDE_OPS_RELU_OP_HPP_
#define TINYTENSOR_INCLUDE_OPS_RELU_OP_HPP_
#include "op.hpp"

namespace TinyTensor{

class ReluOperaor : public Operator
{
private:
    /* data */
    float thresh_ = 0.f;

public:
    ~ReluOperaor() override = default;

    explicit ReluOperaor(float thresh);

    void set_thresh(float thresh);

    float get_thresh() const;
};



}


#endif //TINYTENSOR_INCLUDE_OPS_RELU_OP_HPP_