# CNN-application-on-CRFAR10-picture-distinguish

## 项目简介
[cite_start]本项目是一个基于卷积神经网络 (CNN) 的图像分类系统，旨在对 CIFAR-10 数据集中的 10 类目标（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）进行有效识别与区分 [cite: 180, 224][cite_start]。本项目最初为团队的线性代数课程项目而设计开发 [cite: 4]。

## 环境依赖
在运行本项目之前，请确保您的环境中安装了以下 Python 库：
* Python 3.x
* [cite_start]TensorFlow / Keras [cite: 180]
* [cite_start]NumPy [cite: 223]
* [cite_start]Matplotlib [cite: 180]
* [cite_start]Pillow (PIL) [cite: 223]
* [cite_start]imageio [cite: 238]

## 快速开始

### 1. 数据准备
由于 CIFAR-10 原始数据集格式为二进制文件，在训练前需要将其转换为图像文件。
1. [cite_start]前往多伦多大学官网 (https://www.cs.toronto.edu/~kriz/cifar.html) 下载 CIFAR-10 的 Python 版本压缩包并解压 [cite: 2]。
2. [cite_start]打开 `code/preprocessing_10.py` 文件，将文件中的路径（如 `E:/cifar-10-batches-py/`）修改为您实际解压的路径 [cite: 238]。
3. [cite_start]运行 `preprocessing_10.py`，代码会自动将二进制文件转换为 `.jpg` 格式，并按类别保存在 `train` 和 `test` 文件夹中 [cite: 238, 239]。

### 2. 修改基础路径 (核心注意项)
[cite_start]**重要**：在运行任何训练或测试脚本之前，您必须打开对应的 Python 文件（如 `main.py`, `auto_main.py`, `distinguish.py`, `test_distinguish.py`），并将其中的全局变量 `base_dir` 修改为您本地项目的实际绝对路径 [cite: 3]。

### 3. 模型训练
本项目提供两种训练模式：
* [cite_start]**常规训练**：运行 `code/main.py` [cite: 3][cite_start]。该脚本包含一个标准的 CNN 网络结构，训练完成后会自动在 `mode` 目录下生成带有时间戳和准确率的文件夹，并保存 `.h5` 模型文件和训练过程的 Loss/Accuracy 曲线图 [cite: 236]。
* [cite_start]**自动化超参数调优 (进阶)**：如果您想自动寻找最佳网络参数，请运行 `code/auto_main.py` [cite: 3][cite_start]。运行前请务必在代码开头阅读并设置相关的初始参数（如 `batch_size_num`, `epochs`, `dense_num_max` 等） [cite: 3, 180]。

### 4. 模型测试与预测
您可以提供两种方式来验证模型的准确性：

**方式一：测试互联网上的任意图片**
1. 从互联网上下载待测试的图片（建议为 2009 年之前的图片，或常见的物体图片）。
2. [cite_start]将图片放入项目的 `test/image` 目录中 [cite: 3]。
3. [cite_start]确保 `test/model` 目录下包含您训练好的 `.h5` 模型文件 [cite: 4]。
4. [cite_start]运行 `test/test_distinguish.py`。测试结果（包含图片及其预测的概率分布柱状图）将输出到 `test/res` 目录中 [cite: 4, 241]。

**方式二：使用内部测试集验证**
1. [cite_start]直接运行 `code/distinguish.py` [cite: 4]。
2. [cite_start]脚本会在测试集中随机抽取 10 张图片进行预测，并在 `test_data/res` 目录下生成综合的准确率饼图和每张图片的详细概率分布图 [cite: 224, 228]。

## 版权声明
[cite_start]本项目的代码版权归作者 (Poter & Yang / 账号所有者) 所有 [cite: 5, 245][cite_start]。请勿随意抄袭或用于商业用途 [cite: 6][cite_start]。如有任何问题，欢迎随时联系探讨 [cite: 5]。
