# CIFAR-10 神经网络训练项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

一个基于 CIFAR-10 数据集的深度学习训练框架，支持模型训练、超参数搜索和可视化分析。

## 🚀 功能特性
- ​**模型训练**：使用 `train.py` 统一管理超参数设置，内置数据集加载接口（可自定义）
- ​**训练测试**：使用`test_train.py`执行一次训练，结果保存在目录`test_train`下
- ​**超参数搜索**：使用 `hyperparameter_search.py` 自动优化超参数，结果保存在目录`hyperparameter_search_models`下
- ​**权重可视化**：通过 `weight_plot.py` 生成参数可视化图像，图像保存在目录`test_weight_plot`下
- **训练过程可视化**：通过`train_plot.py`绘制loss和accuracy随训练轮数变化曲线，图像保存在目录`test_train_plot`下
- ​**模型测试**： 通过`test_model.py`在测试已训练模型在测试集上的准确率

## 📂 项目结构
```
.
├── dataset/                       # 数据集根目录
│   ├── cifar-10-batches-py/       # 原始CIFAR-10数据集文件（需自行下载）
│
├── mynn/                          # 自定义神经网络模块
│   ├── __init__.py                # 包初始化
│   ├── activation_function.py     # 激活函数实现
│   ├── initializer.py             # 参数初始化
│   ├── lr_scheduler.py            # 学习率调度器
│   ├── metric.py                  # 评估指标（准确率）
│   ├── models.py                  # 模型定义
│   ├── op.py                      # 基础算子
│   ├── optimizer.py               # 优化器（SGD）
│   └── runner.py                  # 训练流程控制
│
├── hyperparameter_search.py       # 超参数搜索（随机）
├── train.py                       # 主训练脚本
├── test_train.py                  # 训练过程测试
├── train_plot.py                  # 训练指标可视化
├── weight_plot.py                 # 权重可视化工具
├── test_model.py                  # 模型准确率测试
├── requirements.txt               # 依赖库列表
├── LICENSE                        # MIT许可证
├── README.md                      # 项目文档
│
├── hyperparameter_search_models/  # 保存超参数搜索结果
├── test_train/                    # 保存单次训练结果
├── test_train_plot/               # 保存训练图表
└── test_weight_plot/              # 保存权重可视化结果
```
## 🧠 模型说明
模型定义位于 mynn/ 文件夹中，支持自定义网络结构、激活函数和损失函数等。

## 📦 安装指南
### 环境要求
- Python 3.8+ （本项目使用3.10）
- CUDA 12.5 （项目使用 GPU 加速），下载链接 <https://developer.nvidia.com/cuda-12-5-0-download-archive>, 注意需要下载到C盘中，并添加环境变量（如下图所示），否则可能无法正常运行
-  ![环境变量设置](ev_settings.png)
  
### 快速安装
#### 安装依赖
pip install -r requirements.txt

## 🛠 使用说明
- 数据集地址配置，由train.py中model_train()函数下`data_dir`定义（可自行修改）
- 超参数配置范围，具体见train.py中model_train()函数定义
### 训练模型
- 修改超参数：在`test_train.py`文件中根据参数可选范围（见参数配置）修改字典train_params中的超参值
### 超参数搜索
- 修改搜索范围：在`hyperparameter_search.py`文件中修改字典search_config中各个键的值
### 准确率测试
- 测试不同权重文件：修改model.load_model（）中地址的值可测试不同权重文件在测试集下的准确率
### 可视化
- 训练过程可视化：运行`train_plot.py`，超参数修改类同"训练模型"部分
- 权重可视化：运行`weight_plot.py`，需要在该文件的函数visualize_model_params（）中写入正确的权重文件地址（文件类型为.pkl）



