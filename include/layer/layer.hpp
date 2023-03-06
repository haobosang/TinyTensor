/*
 * @Author: haobosang lihaobo1998@gmail.com
 * @Date: 2023-03-06 10:04:15
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-06 17:43:25
 * @FilePath: /MyTinyTensor/include/layer/layer.hpp
 * @Description: 
 */
#ifndef TINYTENSOR_INCLUDE_LAYER_HPP_
#define TINYTENSOR_INCLUDE_LAYER_HPP_

#include "data/Tensor.hpp"
#include <string>

namespace TinyTensor{

class Layer
{
private:
    /* data */
    std::string _layer_name_;

    
public:
    explicit Layer(const std::string &layer_name);
    
    virtual ~Layer() = default;

    virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs);

};



}


#endif //TINYTENSOR_INCLUDE_LAYER_HPP_