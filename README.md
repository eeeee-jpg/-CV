# 计算机视觉实验：图像检索与文本检测联合系统

本项目实现了一个完整的图像检索与文本检测联合实验，包含模型训练、测试、准确率评估与可视化全流程。

> 📺 演示视频（网盘链接）：  
> [https://pan.quark.cn/s/516f2fcbf3a2](https://pan.quark.cn/s/516f2fcbf3a2)

---

## 📁 项目结构
#### 计算机视觉实验/
#### ├── .venv/ # Python 虚拟环境
#### ├── base/ # 图像检索图库
#### │ ├── BJTU/ # 交大相关场景图片
#### │ └── util_pic/ # 无关背景图片
#### ├── data/ # 文本检测训练集
#### │ ├── *.jpg/png # 交大场景图片
#### │ └── *.json # 对应图片的文字标注文件
#### ├── P@K图/ # 检索准确率可视化结果图
#### ├── query/ # 测试查询图片（共 12 个地点）
#### ├── retrieval_results/ # 图像检索结果文件
#### │ └── result_top60.txt # 各查询图的 TOP60 检索结果
#### ├── visual_result/ # 检索 + 文本检测可视化结果（共 24 组）
#### ├── Precision@K.py # 检索准确率计算与绘图程序
#### ├── README.md # 项目说明文档
#### ├── requirements.txt # Python 依赖库列表
#### ├── retrieval_features.pkl # 预训练图像检索特征库
#### ├── retrieval_test.py # 图像检索测试程序（生成 TOP60 结果）
#### ├── retrieval_text_visual.py # 图像检索 + 文本检测可视化程序
#### ├── retrieval_train.py # 图像检索模型训练程序
#### ├── text_det_model.pth # 预训练文本检测模型权重
#### └── text_det_train.py # 文本检测模型训练程序


---

## 🔧 环境依赖配置

### 1. Python 版本
建议使用 **Python 3.8 ~ 3.10**

### 2. 安装依赖库

在项目根目录下执行：

```bash
pip install -r requirements.txt
```
## 🚀 完整运行步骤（按顺序执行）
### 步骤 1：图像检索模块
#### 1.1 训练图像检索特征库
运行 retrieval_train.py：

读取 base/ 文件夹下所有图片

提取 ResNet50 特征

生成 retrieval_features.pkl 特征库文件（仅需运行一次）

```bash
python retrieval_train.py
```
##### 1.2 生成 TOP60 检索结果
运行 retrieval_test.py：

加载特征库

对 query/ 文件夹下的图片进行检索

生成 retrieval_results/result_top60.txt 文件，记录每个查询图的 TOP60 结果

```bash
python retrieval_test.py
```
#### 1.3 计算准确率并绘图
运行 Precision@K.py：

读取 result_top60.txt

统计 12 个地点的 P@20 / P@40 / P@60 平均准确率

在 P@K图/ 文件夹下生成各地点的准确率折线图

```bash
python Precision@K.py
```
### 步骤 2：文本检测模块
#### 2.1 训练文本检测模型
运行 text_det_train.py：

读取 data/ 文件夹下的图片与 JSON 标注文件

训练一个轻量卷积神经网络

生成 text_det_model.pth 模型权重（仅需运行一次）

```bash
python text_det_train.py
```
### 步骤 3：图像检索 + 文本检测联合可视化
运行 retrieval_text_visual.py：

在代码中修改 QUERY_IMG_PATH，指定 query/ 文件夹下的单张查询图片路径

程序自动完成：

对该查询图执行 TOP10 图像检索

对 TOP10 检索图片进行文本检测并画框

结果保存至 visual_result/ 文件夹下以查询图命名的子目录中

按实验要求，从 12 个地点中各选 2 张查询图，累计运行 24 次，即可得到 24 组可视化结果。

```bash
python retrieval_text_visual.py
```
### 📄 文件功能说明
#### 文件 / 文件夹	             说明
#### retrieval_train.py	训练图像检索特征库，生成 .pkl 文件
#### retrieval_test.py	生成 TOP60 检索结果文件
#### Precision@K.py	计算检索准确率并绘制折线图
#### text_det_train.py	训练文本检测模型，生成 .pth 文件
#### retrieval_text_visual.py	检索 + 文本检测联合可视化程序
#### visual_result/	24 组带文字框的检索结果图片
#### P@K图/	12 个地点的检索准确率可视化图表