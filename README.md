<!--
 * @Author: lihaobo
 * @Date: 2023-03-02 10:12:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-15 21:03:54
 * @Description: 请填写简介
-->
# TinyTensor
![Github cmake](https://img.shields.io/badge/cmake-3.16%2B-green)  
TinyTensor是一种用于运行已经训练好的神经网络模型的工具，以便能够使用它们进行各种任务的推理，如图像分类,语义分割等。
![TinyTensor](./img/TinyTensor.png)

## 使用的技术和开发环境
* 开发语言：C++ 17
* 数学库：  Armadillo
* 单元测试：Google Test
## 环境配置

```
apt update
apt install cmake libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```


## 目前已实现的算子
- ReLU
- Sigmod
## 致谢
caffe