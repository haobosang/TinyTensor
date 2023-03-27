<!--
 * @Author: lihaobo
 * @Date: 2023-03-02 10:12:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-27 19:31:20
 * @Description: 请填写简介
-->
# TinyTensor
![Github cmake](https://img.shields.io/badge/cmake-3.16%2B-green)  

![TinyTensor](./img/TinyTensor.png)
TinyTensor supports a variety of popular neural network architectures such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and fully connected neural networks, and it can be used for tasks such as image classification, object detection, speech recognition, and natural language processing.
## Development Environment
* 开发语言：C++ 20
* 数学库：  Armadillo
* 单元测试：Google Test
* 代码风格：Google Style
## How to build on Linux
### Ubuntu 18 (Debian 10)
```
apt update
apt install cmake libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```
### Install Armadillo
```
https://arma.sourceforge.net/docs.html
mkdir build && cd build
cmake ..
make -j8
make install
```

## Operators Currently Implemented
- ReLU
- Sigmod
- Conv
## Acknowledgement
caffe