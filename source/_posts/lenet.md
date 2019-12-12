---
title: LeNet 论文阅读及复现
date: 2019-12-08
mathjax: true
tags:
  - LeNet
  - CNN
---

在入门深度学习时，通常接触到的第一个例子就是 LeNet。LeNet-5 由 Yann LeCun 大神在 1998 年设计「按论文发表时间」， 相较于 VGG 和 Inception 等较大的深度卷积神经网络来说，LeNet 是一个很小的网络，只有 6 万个参数，但是，其奠定了以后很长时间内卷积神经网络的设计模式。

{% asset_img lenet_example.gif [LeNet 检测结果示例] %}

<escape><!-- more --></escape>

遥想 1998 年，笔者刚上小学，第一次接触电脑（县里的扶贫机构向我们学校捐赠了两三台奔腾 IV 处理器的电脑）也是三四年级的时候了，当时的电脑没有连接网络，似乎什么也做不了，老师们只是偶尔在上面玩一玩纸牌游戏。而在 AT&T 内部已经大规模的使用 LeNet 来识别票据了，遗憾的是这么伟大的成果也只能在 AT&T 内部使用，一方面，因为当时信息还不是很发达，不像现在传播的这么快，另一方面，当是几乎没有两家机构使用的是一样的软硬件平台。下面，我们就来剖析一下 LeNet 的设计：

## LeNet 网络结构

LeNet 一共由七个网络层（不包括输入层）组成：**3 个卷积层**、**2 个降采样层**和 **2 个全联接层**，需要注意的是，论文中的**降采样层**与现在常用的**池化层**有一点区别：

- 池化层有平均池化和最大池化两种，计算方式是对池化窗口中的数据求平均或者取最大值，**没有可训练参数**
- 降采样的计算方式是对采样窗口内的数据求和，然后乘以一个**可训练**的系数，并加上一个**可训练**的偏差

LeNet 网络结构如下图所示。

{% asset_img arch_of_lenet.png [LeNet 网络架构] %}

图中，Cx 表示卷积层，Sx 表示降采样层，Fx 表示全联接层，其中 x 表示层数。

### 隐藏层

以下是对隐藏层中每一层的详细说明：

1. C1 是一个卷积层，卷积核大小为 5x5，步长为 1，无填充，6 个输出通道：

   **参数个数**：(5x5+1)x6=156，其中 5x5 为卷积核大小，1 为偏差，6 为输出通道数；

   **连接数**：(5x5+1)x28x28x6=122304，输出 feature map 上每个像素连接到  5x5+1 个参数上，输出一共包含 28x28x6 个像素。 

2. S2 是降采样层，核大小 2x2，步长为 2：

   **参数个数**：(1+1)x6，每层一个系数加一个偏差

   **连接数**：(2x2+1)x14x14x6，输出 feature map 上每个像素连接到 2x2+1 个参数上，一共有 14x14x6 个输出

3. C3 同样是一个卷积层，卷积核大小为 5x5，步长为 1，无填充，16个输出通道。但是，该层与 S2 之间不是全部连接，具体的连接方式如下表所示：

   {% asset_img dropout_of_c3_in_lenet.png [C3 输入与输出映射关系] %}

   图中每一列表示对应层是由 S2 输出通道中的那些层计算得来的，例如：第一列表示，C3 输出的 16 个通道中，第一个通道的结果是由 S2 输出通道 1、2、3 计算得到的。

   **参数个数**：(5x5x3+1)x6+(5x5x4+1)x9+(5x5x6+1)=1516，括号里面的每个数字分别对应（卷积核宽x卷积核高x输入通道数+偏差）

   **连接数**：(5x5x3+1)x6x**10x10**+(5x5x4+1)x9x**10x10**+(5x5x6+1)x**10x10**=151600

4. S4 为降采样层，核大小 2x2，步长 2：

   **参数个数**：(1+1)x16=32

   **连接数**：(2x2+1)x5x5x16=2000

5. C5 为卷积层，卷积核大小 5x5，步长为 1，无填充，120 个输出通道：

   **参数个数**：(5x5x16+1)x120=48120

   **连接数**：与参数个数相同（输出 feature map 的大小为 1x1）

6. F6 全连接层，输入 120，输出 84：

   **参数个数**：(120+1)x84=10164

   **连接数**：(120+1)x84=10164

LeNet 采用了 **sigmoid** 激活函数（squashing function）对上面 C1 到 F6 层的输出进行激活。

> sigmoid 函数的公式是 $sig(x)=\frac{1}{1+e^{-x}}$，作者写的公式是 $f(a)=A\tanh(Sa)$，这是因为 $sig(x)$ 和 $tanh(x)$ 之间可以通过线性变换相互转换，即 $tanh(x)=2sig(2x)-1$，论文中取 S 取 $\frac{2}{3}$，A 取 1.7259。
>
> 所以：$f(a)=1.7259(2sig(\frac{4}{3}x)-1)$ 

### 输出层

最后的输出层由欧式径向基函数（Euclidean Radial Basis Function，RBF）单元组成，关于径向基函数的更多信息可以参考 [wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function) 上的介绍。
$$
y_i=\sum\limits_{j}(x_j-w_{ij})^2
$$

> 关于上式的理解：可以想象输出层就是一个全联接层，不同的是在线性全联接层中的计算公式是 $Y=XW+b$，而此处的计算方式是：用每个输入 $x_j$ 减去其与输出 $y_i$ 之间连线上的权重的平方求和：
>
> {% asset_img rbf.png [RBF 层] %}

那么 $w_{ij}$ 如何取值呢，这就要看下面这张图了（论文图 3），图中有笔迹的地方取 +1，背景取 -1。到这里，相信读者应该可以明白为什么 F6 的输出是 84 了：作者先定义好了每个字母的图案（下图包含所有 ASCII 码，对于 MNIST 数据集只需要取其中的 0～9 就可以了），然后将 C1 到 F6 的计算结果与每个图案进行像素级的对比，如果两者比较接近，那么计算结果 $y_i$ 就会很小，反之则会是一个较大的值。

{% asset_img initial_parameters_of_rbf_for_full_ascii_set.png [RBF 层参数值] %}

## 损失函数

使用于上述网络的最简单的损失函数就是最大似然估计（Maximum Likelihood Estimation，MLE），在这里它等价于最小均方误差。
$$
E(W)=\frac{1}{P}\sum\limits_{p=1}^{P}y_{D_p}(Z^p,W)
$$
式中，$y_{D_p}$是 RBF 第 $D_p$ 个单元的输出，$Z^p$ 为对应的输入，W 为系统的中可调节参数。需要注意的是上式中 $y_{D_p}$ 和 $(Z^p,W)$ 之间不是乘的关系，而是给定输入 $Z^p$ 和系统参数为 $W$ 的情况下，输出为 $y_{D_p}$。

这个损失函数在多数情况下是适用的，但是，它存在下面几个问题：

1. 如果我们允许 RBF 中的参数进行调整，有一个简单但是不可接受（trivial but totally unacceptable）情况可能会出现，即：RBF 层的所有参数向量都相同，并且 F6 的状态与参数向量一致。这种状态下，模型将会忽略输入，并且 RBF 的输出永远为 0。这种状况可以通过固定 RBF 参数来避免；
2. 各个类别之间没有竞争关系，这种竞争关系可以通过更具区别性的训练标准来获得。例如，最大互信息（Maximum Mutual Information）准则。 

## 一些 Tricks

1. MNIST 数据集中图片的大小是 `28x28` 的，为什么 LeNet 中将其扩展为 `32x32`？

   通过这种方式可以使一些潜在的特征（potential distinctive features），如：字符的结尾和拐角等，出现在高层感受野的中心。

2. 为什么要对像素值进行标准化（Normalized）处理？

   LeNet 中将 MNIST 数据集中像素值标准化后，背景色（白色）为 -0.1，前景色（黑色）为 1.175，使得输入数据的均值大致为 0，方差大致为 1，可以加快网络训练的速度。

3. 为什么 S2 到 S3 没有采用全联接？

   1. 降低连接的数量；2. 打破了网络的对称性，使得不同的特征图可以提取不同的特征。

## 算法复现

笔者利用 PyTorch 对 LeNet 进行了复现，与目前多数教程中的**简化版本**（使用平均池化代替降采样，softmax 代替 RBF，随机 Dropout 代替 C3 或完全去除了 Dropout）不同，此实现与论文中描述的模型完全一致，同时，还支持通过参数的方式来指定池化和激活函数，源码已发布到  [GitHub](https://github.com/xinpingwang/paper-with-code/tree/master/lenet)（欢迎 star），也可以直接在 [Google Colab]( https://colab.research.google.com/github/xinpingwang/paper-with-code/blob/master/lenet/lenet.ipynb) 上运行。

>  此外，Wei-Hsiang Wang 有一个仅使用 NumPy 来复现 LeNet 的版本 [LeNet-from-Scratch](https://github.com/mattwang44/LeNet-from-Scratch)（笔者在编码过程中对其进行了一些参考）。

下面是在 BATCH_SIZE 为 32 时，检测正确率随 EPOCH 的变化图。

{% asset_img train.png [训练过程中检测正确率随 Epoch 的变化图] %}

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [LeNet论文阅读：LeNet结构以及参数个数计算](https://blog.csdn.net/silent56_th/article/details/53456522)
- [PyTorch Taipei 2018 week1: LeNet-5](https://mattwang44.github.io/en/articles/PyTorchTP-LeNet/)
- [Andrew Ng interviews Yann LeCun](https://www.youtube.com/watch?v=Svb1c6AkRzE)



