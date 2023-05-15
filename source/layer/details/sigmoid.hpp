/*
 * @Author: lihaobo
 * @Date: 2023-03-09 13:54:14
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-10 10:07:26
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_
#define TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_
#include "layer.hpp"
#include "ops/sigmod_op.hpp"
#include "ops/op.hpp"

namespace TinyTensor
{
    
class SigmodLayer: public Layer
{
private:
    /* data */
     std::shared_ptr<SigmodOperaor> _op;
public:
    explicit SigmodLayer(const std::shared_ptr<Operator> &op);

    ~SigmodLayer() = default;

    void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
};




} // namespace TinyTensor




#endif //TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_