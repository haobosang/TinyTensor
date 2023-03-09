/*
 * @Author: lihaobo
 * @Date: 2023-03-09 13:54:14
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-09 14:10:28
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_
#define TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_
#include "layer.hpp"
#include "ops/sigmod_op.hpp"
namespace TinyTensor
{
    
class sigmod_layer: public Layer
{
private:
    /* data */
     std::shared_ptr<SigmodOperaor> _op;
public:
    sigmod_layer(/* args */);
    ~sigmod_layer();
};




} // namespace TinyTensor




#endif //TINYTENSOR_INCLUDE_SIGMOD_LAYER_HPP_