---
title: AlexNet：开启卷积神经网络的新时代
date: 2019-12-15
mathjax: true
tags:
  - AlexNet
  - CNN
---

在 LeNet 提出后的很长一段时间里，基于神经网络的图像分类算法一度被其他机器学习方法超越，这类图像分类算法使用手工设计的特征提取算法从图像上提取特征，然后使用机器学习模型对图像的特征进行分类。而有些研究者则坚持认为，特征本身也应该是学习得到的，但是很长一段时间里这样的野心都不能实现，这其中可能有许多因素，但最主要的两个就是：**数据**和**硬件**。终于，这两个问题在 2010 年前后得到改善：2009 年，由李飞飞主导的 [ImageNet](http://www.image-net.org/) 数据集诞生了；而且，GPU 也在 2010 年前后开始应用于机器学习。

<escape><!-- more --></escape>

2012 年，在 Florence 举办的那届 ECCV 上，ImageNet 的 Workshop 比较有趣，由于 Hinton 团队的 AlexNet 在 ImageNet 比赛中遥遥领先，所有的人都在等待他的学生 Alex Krizhevsky 的演讲，大家应该可以想象那种万众瞩目的场景。

> 下面这两个链接分别是吴恩达（Andrew Ng）采访 Hinton 时，Hinton 对那个 Workshop 的描述，和 Alex 在 Workshop 上演讲的 PPT：
>
> https://youtu.be/Svb1c6AkRzE?t=1192
>
> http://image-net.org/challenges/LSVRC/2012/supervision.pdf

下面，我们一起来研究一下将卷积神经网络带上「神坛」的  AlexNet 是如何设计的。

## 网络结构

{% asset_img alexnet_arch.png [AlexNet 网络结构] %}

上图是 AlexNet 的网络结构，其参数数量超多了 6 千万个，是 LeNet 1000 倍。但是，从总体上看，AlexNet 与 LeNet 还是比较相似，都是先通过卷积神经网络提取特征，然后，由多层感知机根据特征对图像进行分类，所以，论文中作者只花了一点篇幅来描述网络中每一层的输入输出，将重点放在如何加快模型的训练速度和降低过拟合。

需要注意的一点是，在 AlexNet 中采用了 **Overlapping Pooling**：在 LeNet 中，降采样层的步长和核大小是一致的，也就是说在滑动过程中不会有重叠的部分，AlexNet 中步长小于池化核的大小，作者认为这样可以减少池化过程中信息的损失量。

## 提升训练速度

### ReLU

对神经元建模的标准形式是 sigmoid，即 $f(x)=tanh(x)$ 或者 $f(x)=(1+e^{-x})^{-1}$，但是，从训练时间的角度来说，这些**饱和**（saturating）非线性函数比**不饱和**（non-saturating）非线性函数 （如：ReLU，$f(x)=max(0,x)$ ）慢许多，尤其是在深度卷积神经网络上。

{% asset_img relu_vs_tanh.png [ReLU 和 tanh 训练速度对比] %}

上图是分别使用 ReLU 和 tanh 作为激活函数时，一个 4 层卷积神经网络在 CIFAR-10 上训练速度的对比，从图中可见，使用 ReLU 作为激活函数的模型快了将近 6 倍。

### 多 GPU 训练

由于 AlexNet 中的参数过多，而 GTX 580 GPU 仅有 3GB 的显存，AlexNet 采用两张 GPU 分两路运算。为此，作者还实现了针对 GPU 高度优化的二维卷积和其他训练中所必需的操作。相关代码作者已经开源，可以在 Google Code 上获取：http://code.google.com/p/cuda-convnet/。

如果将 GPU 内存足够，可以将 AlexNet 合为一路，这时，网络的结构将是下图这样：

{% asset_img alexnet_all_in_one.png [AlexNet 网络结构] %}

## 降低过拟合

### 数据增强

对于图像数据，通过标签不变的变换来扩增数据集是最简单和常用的降低模型过拟合的方式。论文使用了如下两种方法来增加数据

1. 翻转和裁剪（Reflection & Extracting）

   **训练时：** 从预处理得到的 256x256 的图像上依次切出 1024 张 224x224 的图，将这些图及它们的水平翻转作为训练数据，这使得训练数据集扩增了 2048 倍。

   **测试时：** 从测试图像上提取 5 张 224x224 的图（4 个角，以及正中间），以及它们的水平翻转，取模型对这 10 张图输出的 softmax 的平均值作为对原图的判断结果。

2. PCA 

   首先，在整个 ImageNet 数据集上，对图像的 RGB 像素值集进行主成分分析；然后，对每张训练图像添加多个主成分：大小与相应的特征值*乘以均值为 0 且标准偏差为 0.1 的高斯随机变量*成比例。即：
$$
   I_{xy}^\prime=[I_{xy}^R,I_{xy}^G,I_{xy}^B] + [p_1,p_2,p_3][\alpha_1\lambda_1,\alpha_2\lambda_2,\alpha_2\lambda_2]
$$
   式中，$p_i$ 和 $\lambda_i$ 是该像素值的 3x3 协方差矩阵的第 $i$ 个特征向量和特征值，$\alpha_i$ 是前述的随机变量。这个变换体现了自然图像的一个重要属性，即：图像的标签，在不同亮度和颜色下是不变的，降低了 1% 以上的 top-1 错误率。

### 局部响应归一化

作者发现，下面的局部响应归一化有利于模型的泛化，使得 AlexNet 在 ImageNet 数据集上的 top-1 和 top-5 错误率分别降低了 1.4% 和 1.2%，此外，作者还在 CIFAR-10 数据集上的测试了该方法的有效性，对比没有使用归一化的四层 CNN，测试错误率从 13% 降低到了 11%。
$$
b_{x,y}^i=a_{x,y}^i/\left(k+\alpha\sum\limits_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(a_{x,y}^j)^2\right)^\beta
$$
式中：

- $a_{x,y}^i$ 是 $x,y$ 位置上原本的卷积输出
- $b_{x,y}^i$ 是 LRN 的输出值，作为下一层的输入
- N 是该层卷积核的通道数
- n 是要做加总运算的邻近卷积核数量，是一个超参数
- $k,\alpha,\beta$ 均是超参数，论文中选用的是 $k=2,\alpha=10^{-4},\beta=0.75$。

公式中看起来比较复杂的求和，其实就是将临近 n 个 feature maps 上相同位置的值做平方和，示意图如下：

{% asset_img lrn.gif [LRN 计算示意图] %}

> 图片来自，WEI-HSIANG WANG 的博客 https://mattwang44.github.io/en/articles/PyTorchTP-AlexNet/，本文中的内容也对其进行了较多的参考，在此表示感谢。

### Dropout

所谓 Dropout 就是按照一定的概率（0.5）将「神经元」的输出置为 0，这部分输出置 0 的「神经元」在前向传播中对于整个模型的输出来说没有任何贡献，并且也不会参与反向传播。那么这部分神经元存在的价值是什么呢，或者说为什么 Dropout 可以降低模型的过拟合？

- 将多个模型的预测结果结合起来作出判断，是一个非常成功的降低测试错误率的方法，但是，对于一个需要好多天才能完成训练的大型网络来说，显得成本比较高。Dropout 相对而言就是一个非常高效的方式了：在训练过程中，按照 0.5 的概率将模型中「神经元」的输出置 0，相当于每次只有一半的模型参与了预测；而在测试中，整个模型都会参与预测，整个模型就相当于两个小模型的结合。
- 还有一种解释是，训练时模型中的每个「神经元」的输出都有可能置 0，可以避免模型过多的依赖某个（些）「神经元」的输出。

AlexNet 模型中的前两个全联接层加上了 Dropout，有效避免了过拟合，但是，也使得收敛所需的迭代次数增加了 1 倍。

## 训练细节

### 参数初始化

- 权重：均值为 0 标准差为 0.01 的高斯分布
- 偏差：
  - 1 (at 2nd, 4th, 5th and fc layers), 使得训练的前期，ReLU 的输入为正值，加速模型的训练； 
  - 0 (at other layers)

### 随机梯度下降

```python
batch_size=128, momentum=0.9, weight_decay=0.00005
learning_rate=0.01 #若测试错误率不再下降则除以 10
```

权重 $w_i$ 的更新规则：
$$
\begin{align}
v_{i+1} &:= 0.9 \cdot v_i-0.0005 \cdot \epsilon \cdot w_i-\epsilon \cdot \left\langle\frac{\partial L}{\partial w}|\_{w_i}\right\rangle_{D_i} \\\\
w_{i+1} &:= w_i+v_{i+1}
\end{align}
$$

## 算法复现

同样的，笔者使用 PyTorch 对 AlexNet 进行了复现，源码发布在 [GitHub](https://github.com/xinpingwang/paper-with-code/tree/master/alexnet) 上。

## 参考资料

- [ImageNet Classiﬁcation with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Alex Krizhevsky 在 ImageNet Workshop, ECCV 2012 上的演讲](http://image-net.org/challenges/LSVRC/2012/supervision.pdf)
- [吴恩达（Andrew Ng）对 Hinton 的专访](https://youtu.be/Svb1c6AkRzE?t=1192)
- [PyTorch Taipei 2018 week1: AlexNet](https://mattwang44.github.io/en/articles/PyTorchTP-AlexNet/)

