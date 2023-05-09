<!--
 * @Author: lihaobo
 * @Date: 2023-03-02 10:12:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-27 20:08:14
 * @Description: 请填写简介
-->
# TinyTensor
![Github cmake](https://img.shields.io/badge/cmake-3.16%2B-green)  

![TinyTensor](./img/TinyTensor.png)
TinyTensor supports a variety of popular neural network architectures such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and fully connected neural networks, and it can be used for tasks such as image classification, object detection, speech recognition, and natural language processing.
## Development Environment
* Development   language: C++ 20
* Math Library: Armadillo
* Logging framework：Google glog
* unit test:    Google Test
* Code style:   Clang format
* performance testing： Benckmark

## How to build on Linux
### Ubuntu 18 (Debian 10)
```
apt update
apt install cmake libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```
### Install Armadillo
```
wget https://sourceforge.net/projects/arma/files/armadillo-12.2.0.tar.xz
mkdir build && cd build
cmake ..
make -j8
make install
```

## Operators Currently Implemented
* ReLU
* Sigmod
* Conv
## Acknowledgement
caffe