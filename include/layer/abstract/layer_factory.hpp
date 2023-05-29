#ifndef TINYTENSOR_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define TINYTENSOR_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_

#include "layer.hpp"
#include "runtime/runtime_op.hpp"
#include <map>
#include <memory>
#include <string>

namespace TinyTensor
{
    class LayerRegisterer
    {
    public:
        typedef ParseParameterAttrStatus (*Creator)(const std::shared_ptr<RuntimeOperator> &op,  std::shared_ptr<Layer> &layer);

        typedef std::map<std::string, Creator> CreateRegistry;

        static void RegisterCreator(const std::string &layer_type, const Creator &creator);

        static std::shared_ptr<Layer>
        CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

        static CreateRegistry &Registry();
    };

    class LayerRegistererWrapper
    {
    public:
        LayerRegistererWrapper(const std::string &layer_type, const LayerRegisterer::Creator &creator)
        {
            LayerRegisterer::RegisterCreator(layer_type, creator);
        }
    };

} // namespace TinyTensor

#endif // TinyTensor_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_