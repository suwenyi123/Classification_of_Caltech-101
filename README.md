# 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

## **1.** 数据准备

[Caltech-101]数据集的链接`https://data.caltech.edu/records/mzrjq-6wc02`
- 文件`split_data.py`:将原始数据集按照类别分割为80%的训练集和20的测试集

- 原始数据集结构 
dataset/
└── images/
    └── 101_ObjectCategories/
        |── 类别1/
        |   ├── 图片1.jpg
        │   ├── 图片2.jpg
        │   └── ...
        ├── 类别2/
        │   ├── 图片1.jpg
        │   ├── 图片2.jpg
        │   └── ...
        └── ...

- 拆分后数据集结构
dataset/
└── images/
    ├── train/
    │   ├── 类别1/
    │   │   ├── 训练图片1.jpg
    │   │   ├── 训练图片2.jpg
    │   │   └── ...
    │   ├── 类别2/
    │   │   ├── 训练图片1.jpg
    │   │   ├── 训练图片2.jpg
    │   │   └── ...
    │   └── ...
    ├── test/
    │   ├── 类别1/
    │   │   ├── 测试图片1.jpg
    │   │   ├── 测试图片2.jpg
    │   │   └── ...
    │   ├── 类别2/
    │   │   ├── 测试图片1.jpg
    │   │   ├── 测试图片2.jpg
    │   │   └── ...
    │   └── ...
    └── 101_ObjectCategories/
        ├── 类别1/
        │   ├──
        │   └── ...
        └── ...

# **2.** 训练与测试

本实验通过修改**ResNet-18、ResNet-50、ResNet-101、EfficientNet-B4**架构在Caltech 101图像分类数据集上的性能表现。实验主要包含两个部分：首先通过网格搜索对超参数进行优化，找到最佳的模型配置；然后对比预训练模型与随机初始化模型的性能差异，验证迁移学习在图像分类任务中的有效性。实验过程中使用 TensorBoard对训练过程进行可视化，记录训练集和验证集的损失函数变化以及验证集准确率的变化趋势。

- `main_ResNet-18.py` : 包括baseline模型初始化和pretrained ResNet-18模型生成的模型权重会以pth的形式自动保存；训练中产生的loss和Accuracy信息会记录在tensorboard中

- `main_ResNet-50.py`:pretrained ResNet-50模型生成的模型权重会以pth的形式自动保存；训练中产生的loss和Accuracy信息会记录在tensorboard中

- `main_ResNet-101.py`:pretrained ResNet-50模型生成的模型权重会以pth的形式自动保存；训练中产生的loss和Accuracy信息会记录在tensorboard中

- `main_Efficient_B4.py`:pretrained Efficeient-B4模型生成的模型权重会以pth的形式自动保存；训练中产生的loss和Accuracy信息会记录在tensorboard中

| 模型结构       | batch_size | epochs | lr    | weight_decay | 解冻策略         | Top@1  |
|----------------|------------|--------|-------|--------------|------------------|--------|
| baseline       | 64         | 50     | 3e-4  | 1e-5         | layer3+layer4    | 69.44  |
| ResNet-18      | 64         | 30     | 3e-4  | 1e-5         | layer4           | 91.81  |
| ResNet-50      | 32         | 30     | 1e-3  | 1e-5         | layer4           | 93.56  |
| ResNet-101     | 32         | 50     | 5e-4  | 1e-5         | layer3           | 94.18  |
| EfficientNet-B4| 64         | 30     | 3e-4  | 1e-4         | last 6 layers    | 97.57  |

验证了迁移学习对小样本分类任务的必要性以及深层网络对预训练特征的利用更充分

## 可视化结果
使用tensorboard进行acc，loss等指标的可视化
- `tensorboard --logdir=runs/  --host=127.0.0.1 --port=1111`
- `tensorboard --logdir=runs/  --host=127.0.0.1 --port=2222`

会在网页端显可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化