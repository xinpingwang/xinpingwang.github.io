---
title: 第一行代码
date: 2019-10-22
tags:
 - machine learning framework
---

今天终于要正式开始构建机器学习框架了👏，在目前最为主流的神经网络框架中 TensorFlow 有 300 多万行代码，PyTorch 的代码量相对少一些但也有 80 多万行，显然以个人能力不可能构建出一个如此复杂的工程，甚至以笔者目前的工程能力和学术水平都不足以完成一个小型神经网络框架的设计与构建。因此，我选择了一个相对较小的神经网络框架 [decaf](https://github.com/Yangqing/decaf.git) ~~作~~（jing）~~为~~（xing）~~参~~（chao）~~考~~（xi），该项目是贾扬清在 2013 年（7 月——9 月）完成的一个项目，最终版本代码量在 1 万行左右，根据 GitHub 上的项目简介，该项目是 Caffe 的前身，目标是实现一个高效而灵活的卷积神经网络框架。

<escape><!-- more --></escape>

## 一个例子

在开始具体的设计之前，我们先看一下使用现在主流的神经网络框架之一 —— PyTorch 时，是如何实现一个网络的：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
```

这段代码来自 PyTorch 基础教程 [Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)，实现了著名的 **LeNet**，可以说非常的 Python。我们从机器学习架设计的角度来看一下这段代码：

1. 在 `__init__` 函数里，用到了 `nn.Linear` 和 `nn.Conv2d`，这两个网络层是由 PyTorch 为我们提供的，所以，我们也需要实现一些常见的网络层，供用户使用；
2. 定义了 `forward` 函数来处理正向传播的计算，但是，却没有处理如何进行反向传播，这意味着我们的神经网络框架要自动根据用户定义的正向传播计算过程，来构建计算图，并根据计算图实现反向传播。

## 三个目标

通过上面两点，我们基本可以明确我们在本机器学习框架中需要完成的事情：

1. 实现一些常见的网络层，例如：Linear、Conv2d，以及一些常用的激活函数 ReLU、Sigmoid，同时，还要提供相应的反向传播（自动求导）计算实现；
2. 实现计算图的构建；
3. 实现一些常用的损失函数和优化算法。

## 框架设计

针对这几个目标，需要抽象出一些数据结构和接口。主要有 Blob、Layer 和 Net。

### Blob

根据 [前向传播与反向传播](https://xinpingwang.github.io/2019/10/15/forward-and-backward/) 中的描述，在训练的过程中，需要同时保留参数的当前值和梯度。我们将这两个数据放在一起组成一个 `Blob`：

```python
class Blob(object):
    
    def __init__(self, shape=None, dtype=None):
        if shape is None and dtype is None:
            self._data = None
        else:
            self._data = np.zeros(shape, dtype=dtype)
        self._diff = None
```

其中 `_data` 和 `_diff` 都是 NdArray。

### Layer

`Layer` 是对不同网络层的抽象，定义了 `forward` 和 `backward` 函数，需要在子类中进行实现。模型被看成一个从下往上算的过程，所以使用 `bottom` 和 `top` 来代表每一个 Layer 的输入和输出。

```python
class Layer(object):
    
    def __init__(self, **kwargs):
        self.spec = kwargs
        self.name = self.spec['name']
        self._param = []

    def forward(self, bottom, top):
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def param(self):
        return self._param
```

### Net

`Net` 表示一个具体的模型，由一个或多个 `Layer` 组成，并根据 `Layer` 来构建出计算图，实现相对复杂一些，后面会转门写一节来介绍 Net 的实现。

```python
class Net(object):

    def __init__(self):
        self._graph = nx.DiGraph()
        self._blobs = defaultdict(Blob)
        self._layers = {}
        self._needs = {}
        self._provides = {}
        # The topological order to execute the layer.
        self._forward_order = None
        self._backward_order = None

    def add_layer(self, layer, needs=[], provides=[]):
        pass

    def execute(self):
        # the forward pass. we will also accumulate the loss function
        loss = 0.
        for _, layer, bottom, top in self._forward_order:
            loss += layer.forward(bottom, top)
        # the backward pass
        for _, layer, bottom, top, propagate_down in self._backward_order:
            layer.backward(bottom, top, propagate_down)
        return loss

    def update(self):
        for _, layer, _, _ in self._forward_order:
            layer.update()
```





