/*
 * @Author: lihaobo
 * @Date: 2023-03-06 09:56:52
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-09 10:18:22
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define TINYTENSOR_INCLUDE_FACTORY_LAYER_FACTORY_HPP_

#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace TinyTensor{

class LayerRegisterer {
public:
    typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

    typedef std::map<OpType,Creator> CreateRegistry;

    static void RegisterCreator(OpType op_type,const Creator &creator);

    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

    static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};

} // namespace TinyTensor


#endif //TINYTENSOR_INCLUDE_FACTORY_LAYER_FACTORY_HPP_