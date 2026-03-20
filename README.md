# CNN-application-on-CRFAR10-picture-distinguish

## Project Introduction
This project is an image classification system based on a Convolutional Neural Network (CNN), designed to effectively identify and distinguish 10 categories of objects in the CIFAR-10 dataset (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This project was originally designed and developed for the team's linear algebra course project.

## Dependencies
Before running this project, please ensure the following Python libraries are installed in your environment:
* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Pillow (PIL)
* imageio

## Quick Start

### 1. Data Preparation
Since the original CIFAR-10 dataset is in binary format, it needs to be converted into image files before training.
1. Go to the University of Toronto website (https://www.cs.toronto.edu/~kriz/cifar.html) to download and unzip the Python version of CIFAR-10.
2. Open the `code/preprocessing_10.py` file and modify the path (e.g., `E:/cifar-10-batches-py/`) to your actual unzipped path.
3. Run `preprocessing_10.py`, and the code will automatically convert the binary files into `.jpg` format and save them in the `train` and `test` folders by category.

### 2. Modify Base Directory (Core Requirement)
**Important**: Before running any training or testing scripts, you must open the corresponding Python file (e.g., `main.py`, `auto_main.py`, `distinguish.py`, `test_distinguish.py`) and change the global variable `base_dir` to the actual absolute path of your local project[cite: 3].

### 3. Model Training
This project provides two training modes:
* **Regular Training**: Run `code/main.py`. This script contains a standard CNN architecture. After training, it will automatically generate a folder with a timestamp and accuracy under the `mode` directory, saving the `.h5` model file and the Loss/Accuracy curve graphs of the training process.
* **Automated Hyperparameter Tuning (Advanced)**: If you want to automatically find the best network parameters, please run `code/auto_main.py`. Before running, be sure to read the beginning of the code and set the relevant initial parameters (such as `batch_size_num`, `epochs`, `dense_num_max`, etc.).

### 4. Model Testing and Prediction
You can verify the accuracy of the model in two ways:

**Method 1: Test any image from the internet**
1. Download the image to be tested from the internet (images before 2009 or common object images are recommended).
2. Place the image into the project's `test/image` directory.
3. Ensure that the `test/model` directory contains your trained `.h5` model file.
4. Run `test/test_distinguish.py`. The test results (including the image and its predicted probability distribution bar chart) will be output to the `test/res` directory.

**Method 2: Verify using the internal test set**
1. Run `code/distinguish.py` directly.
2. The script will randomly select 10 images from the test set for prediction, and generate a comprehensive accuracy pie chart and detailed probability distribution charts for each image in the `test_data/res` directory.

## Copyright Statement
The copyright of the code in this project belongs to the authors (Poter & Yang / account owner). Please do not plagiarize or use it for commercial purposes. If you have any questions, feel free to contact us for discussion.

---

# CNN-application-on-CRFAR10-picture-distinguish

## 项目简介
本项目是一个基于卷积神经网络 (CNN) 的图像分类系统，旨在对 CIFAR-10 数据集中的 10 类目标（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）进行有效识别与区分。本项目最初为团队的线性代数课程项目而设计开发。

## 环境依赖
在运行本项目之前，请确保您的环境中安装了以下 Python 库：
* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Pillow (PIL)
* imageio

## 快速开始

### 1. 数据准备
由于 CIFAR-10 原始数据集格式为二进制文件，在训练前需要将其转换为图像文件。
1. 前往多伦多大学官网 (https://www.cs.toronto.edu/~kriz/cifar.html) 下载 CIFAR-10 的 Python 版本压缩包并解压
2. 打开 `code/preprocessing_10.py` 文件，将文件中的路径（如 `E:/cifar-10-batches-py/`）修改为您实际解压的路径
3. 运行 `preprocessing_10.py`，代码会自动将二进制文件转换为 `.jpg` 格式，并按类别保存在 `train` 和 `test` 文件夹中

### 2. 修改基础路径 (核心注意项)
**重要**：在运行任何训练或测试脚本之前，您必须打开对应的 Python 文件（如 `main.py`, `auto_main.py`, `distinguish.py`, `test_distinguish.py`），并将其中的全局变量 `base_dir` 修改为您本地项目的实际绝对路径

### 3. 模型训练
本项目提供两种训练模式：
* **常规训练**：运行 `code/main.py`该脚本包含一个标准的 CNN 网络结构，训练完成后会自动在 `mode` 目录下生成带有时间戳和准确率的文件夹，并保存 `.h5` 模型文件和训练过程的 Loss/Accuracy 曲线图
* **自动化超参数调优 (进阶)**：如果您想自动寻找最佳网络参数，请运行 `code/auto_main.py`。运行前请务必在代码开头阅读并设置相关的初始参数（如 `batch_size_num`, `epochs`, `dense_num_max` 等）。

### 4. 模型测试与预测
您可以提供两种方式来验证模型的准确性：

**方式一：测试互联网上的任意图片**
1. 从互联网上下载待测试的图片（建议为 2009 年之前的图片，或常见的物体图片）。
2. 将图片放入项目的 `test/image` 目录中
3. 确保 `test/model` 目录下包含您训练好的 `.h5` 模型文件
4. 运行 `test/test_distinguish.py`。测试结果（包含图片及其预测的概率分布柱状图）将输出到 `test/res` 目录中

**方式二：使用内部测试集验证**
1. 直接运行 `code/distinguish.py`
2. 脚本会在测试集中随机抽取 10 张图片进行预测，并在 `test_data/res` 目录下生成综合的准确率饼图和每张图片的详细概率分布图

## 版权声明
本项目的代码版权归作者 (Poter & Yang / 账号所有者) 所有。请勿随意抄袭或用于商业用途。如有任何问题，欢迎随时联系探讨。
