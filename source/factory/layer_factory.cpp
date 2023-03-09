/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:07:18
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-09 10:42:24
 * @Description: 请填写简介
 */

#include "glog/logging.h"
#include "factory/layer_factory.hpp"

namespace TinyTensor{


void LayerRegisterer::RegisterCreator(OpType op_type,const Creator &creator){
    CHECK(creator != nullptr) <<"Layer creator is empty";
    CreateRegistry &registry = Registry();
    CHECK_EQ(registry.count(op_type),0) << "Layer type: " << int(op_type) << " has already registered!";
    registry.insert({op_type,creator});
    return ;
}

std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op){
    CreateRegistry &registry = Registry();

    const OpType op_type = op->op_type_;
    LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);

    const auto &creator = registry.find(op_type)->second;
    LOG_IF(FATAL, !creator) << "Layer creator is empty!";

    std::shared_ptr<Layer> layer = creator(op);
     LOG_IF(FATAL, !layer) << "Layer init failed!";
    return layer;



}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry(){
    static CreateRegistry *Registry = new CreateRegistry();
    CHECK(Registry != nullptr);
    return *Registry;
}

}