# 前言

## 基础环境

为简单安装步骤，保持一个干净的运行环境，使用docker进行部署。

- 在[Docker Hub](https://hub.docker.com/)里找一个cuda的最新镜像
- 本次测试使用[nvidia](https://hub.docker.com/r/nvidia/cuda/)的官方驱动，tag == 8.0-cudnn7-devel
- `docker pull nvidia/cuda:8.0-cudnn7-devel`
- `docker run -it nvidia/cuda:8.0-cudnn7-devel bash`

## 安装python, pytorch

- `apt-get update && apt-get install python`
- 在[pytorch官方](http://pytorch.org/)下载对应pytorch包，此处使用pip
- `pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl`
- `pip install torchvision`

安装完成后torch路径应在 /usr/local/lib/python2.7/dist-packages/torch 中

之后分析以此代码为准