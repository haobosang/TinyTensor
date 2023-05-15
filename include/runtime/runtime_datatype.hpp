/*
 * @Author: lihaobo
 * @Date: 2023-03-11 13:46:43
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-11 14:32:01
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
#define TINYTENSOR_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
/// 计算节点属性中的权重类型
enum class RuntimeDataType {
  kTypeUnknown = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};
#endif // TINYTENSOR_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
