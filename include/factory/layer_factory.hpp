#ifndef TINYTENSOR_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define TINYTENSOR_INCLUDE_FACTORY_LAYER_FACTORY_HPP_

#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace TinyTensor{

class LayerRegisterer {
public:
    typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

    typedef std::map<OpType,Creator> CreateRegistry;

    static void RegisterCreator(OpType op_type,const Creator &creater);

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