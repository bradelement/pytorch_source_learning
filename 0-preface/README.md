# 前言

## 安装环境

为简单安装步骤，保持一个干净的运行环境，使用docker进行部署。

- 在[Docker Hub](https://hub.docker.com/)里找一个cuda的最新镜像。
- 本次测试使用[nvidia的官方驱动](https://hub.docker.com/r/nvidia/cuda/)，tag == 9.1-cudnn7-devel
- `docker pull nvidia/cuda:9.1-cudnn7-devel`