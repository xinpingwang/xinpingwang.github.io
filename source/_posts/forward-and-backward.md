---
title: 前向传播与反向传播
date: 2019-10-15
mathjax: true
tag:
  - machine learning framework
---

> 本文主要内容参考自「动手学深度学习」中的**正向传播、反向传播和计算图**一节，略作修改。

在利用机器学习框架来实现模型时，我们只需要提供模型的正向传播（forward propagation）的计算，即如何根据输⼊计算模型输出，框架的自动求导模块会⾃动⽣成反向传播（back-propagation）计算中用到的梯度，我们不再需要根据链式法则去手动推倒梯度。可以说框架为我们提供的⾃动求梯度极⼤简化了深度学习模型训练算法的实现，降低了机器学习入门的门槛。

本文中我们将使⽤数学和计算图（computational graph）两个⽅式来描述一个**带 $L_{2}$ 范数正则化的含单隐藏层的多层感知机模型**的正向传播和反向传播，为接下来动手实现机器学习框架做好准备。

<!-- more -->

## 正向传播

正向传播是指对神经⽹络沿着从输⼊层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。为简单起⻅，假设输⼊是⼀个特征为 $\boldsymbol{x} \in \mathbb{R}^{d}$ 的样本，且不考虑偏差项，那么中间变量
$$
\boldsymbol{z}=\boldsymbol{W}^{(1)} \boldsymbol{x}
$$
其中 $\boldsymbol{W}^{(1)} \in \mathbb{R}^{h \times d}$ 是隐藏层的权重参数。把中间变量 $\boldsymbol{z} \in \mathbb{R}^{h}$ 输⼊按元素运算的激活函数 $\phi$ 后，将得到向量⻓度为 $h$ 的隐藏层变量
$$
\boldsymbol{h}=\phi(\boldsymbol{z})
$$
隐藏层变量 $h$ 也是⼀个中间变量。 假设输出层参数只有权重 $\boldsymbol{W}^{(2)} \in \mathbb{R}^{q \times h}$ 的输出层变量，可以得到向量⻓度为 $q$ 的输出层变量
$$
\boldsymbol{o}=\boldsymbol{W}^{(2)} \boldsymbol{h}
$$
假设损失函数为 $\ell$，且样本标签为 $y$，可以计算出单个数据样本的损失项
$$
L=\ell(\boldsymbol{o}, y)
$$
根据 $L_{2}$ 范数正则化的定义，给定超参数 $\lambda$，正则化项即
$$
s=\frac{\lambda}{2}\left(\left\|\boldsymbol{W}^{(1)}\right\|\_{F}^{2}+\left\|\boldsymbol{W}^{(2)}\right\|\_{F}^{2}\right)
$$
其中矩阵的 [Frobenius 范数](http://mathworld.wolfram.com/FrobeniusNorm.html) 等价于将矩阵变平为向量后计算 $L_{2}$ 范数。最终，模型在给定的数据样本上带正则化的损失为
$$
J=L+s
$$
我们将 $J$ 称为有关给定数据样本的⽬标函数，并在以下的讨论中简称⽬标函数。

## 正向传播的计算图

通过绘制计算图可以可视化运算符和变量在计算中的依赖关系，下图为上述模型正向传播的计算图，其中左下⻆是输⼊，右上⻆是输出。可以看到，图中箭头⽅向⼤多是向右和向上，其中⽅框代表变量，圆圈代表运算符，箭头表⽰从输⼊到输出之间的依赖关系。

{% asset_img forward.svg [正向传播计算图] %}

## 反向传播

反向传播指的是计算神经⽹络参数梯度的⽅法。总的来说，反向传播依据微积分中的链式法则，沿着从输出层到输⼊层的顺序，依次计算并存储⽬标函数有关神经⽹络各层的中间变量以及参数的梯度。对输⼊或输出 $X, Y, Z$ 为任意形状张量的函数 $Y=f(X)$ 和 $Z=g(Y)$，通过链式法则，我们有
$$
\frac{\partial \mathrm{Z}}{\partial \mathrm{X}}=\operatorname{prod}\left(\frac{\partial \mathrm{Z}}{\partial \mathrm{Y}}, \frac{\partial \mathrm{Y}}{\partial \mathrm{X}}\right)
$$
其中 prod 运算符将根据两个输⼊的形状，在必要的操作（如转置和互换输⼊位置）后对两个输⼊做乘法。

本文中示例模型的参数是 $\boldsymbol{W}^{(1)}$ 和 $\mathbf{W}^{(2)}$， 因此反向传播的⽬标是计算 $\partial J / \partial \boldsymbol{W}^{(1)}$ 和 $\partial J / \partial \boldsymbol{W}^{(2)}$ 。 我们将应⽤链式法则依次计算各中间变量和参数的梯度， 其计算次序与前向传播中相应中间变量的计算次序恰恰相反。⾸先，分别计算⽬标函数 $J=L+s$ 有关损失项 $L$ 和正则项 $s$ 的梯度
$$
\frac{\partial J}{\partial L}=1, \quad \frac{\partial J}{\partial s}=1
$$
其次，依据链式法则计算⽬标函数有关输出层变量的梯度 $\partial J / \partial \boldsymbol{o} \in \mathbb{R}^{q}$：
$$
\frac{\partial J}{\partial o}=\operatorname{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial o}\right)=\frac{\partial L}{\partial o}
$$
接下来，计算正则项有关两个参数的梯度：
$$
\frac{\partial s}{\partial \boldsymbol{W}^{(1)}}=\lambda \boldsymbol{W}^{(1)}, \quad \frac{\partial s}{\partial \boldsymbol{W}^{(2)}}=\lambda \boldsymbol{W}^{(2)}
$$
现在，我们可以计算最靠近输出层的模型参数的梯度 $\partial J / \partial \boldsymbol{W}^{(2)} \in \mathbb{R}^{q \times h}$ 。依据链式法则，得到
$$
\frac{\partial J}{\partial \boldsymbol{W}^{(2)}}=\operatorname{prod}\left(\frac{\partial J}{\partial \boldsymbol{o}}, \frac{\partial \boldsymbol{o}}{\partial \boldsymbol{W}^{(2)}}\right)+\operatorname{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \boldsymbol{W}^{(2)}}\right)=\frac{\partial J}{\partial \boldsymbol{o}} \boldsymbol{h}^{\top}+\lambda \boldsymbol{W}^{(2)}
$$
沿着输出层向隐藏层继续反向传播，隐藏层变量的梯度 $\partial J / \partial h \in \mathbb{R}^{h}$ 可以这样计算：
$$
\frac{\partial J}{\partial \boldsymbol{h}}=\operatorname{prod}\left(\frac{\partial J}{\partial \boldsymbol{o}}, \frac{\partial \boldsymbol{o}}{\partial \boldsymbol{h}}\right)=\boldsymbol{W}^{(2)} \frac{\partial J}{\partial \boldsymbol{o}}
$$


由于激活函数 $\phi$ 是按元素运算的，中间变量 $z$ 的梯度 $\partial J / \partial z \in \mathbb{R}^{h}$  的计算需要使⽤按元素乘法符⊙：
$$
\frac{\partial J}{\partial z}=\operatorname{prod}\left(\frac{\partial J}{\partial h}, \frac{\partial h}{\partial z}\right)=\frac{\partial J}{\partial h} \odot \phi^{\prime}(z)
$$
最终，我们可以得到最靠近输⼊层的模型参数的梯度 $\partial J / \partial \boldsymbol{W}^{(1)} \in \mathbb{R}^{h \times d}$ 。依据链式法则，得到

$$
\frac{\partial J}{\partial \boldsymbol{W}^{(1)}}=\operatorname{prod}\left(\frac{\partial J}{\partial z}, \frac{\partial z}{\partial \boldsymbol{W}^{(1)}}\right)+\operatorname{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \boldsymbol{W}^{(1)}}\right)=\frac{\partial J}{\partial \boldsymbol{z}} \boldsymbol{x}^{\top}+\lambda \boldsymbol{W}^{(1)}
$$

## 训练深度学习模型

在训练深度学习模型时，正向传播和反向传播之间相互依赖：

⼀⽅⾯，正向传播的计算可能依赖于模型参数的当前值，而这些模型参数是在反向传播的梯度计算后通过优化算法迭代的。例如，计算正则化项 $s=(\lambda /2)\left(\left\|\boldsymbol{W}^{(1)}\right\|\_{F}^{2}+\left\|\boldsymbol{W}^{(2)}\right\|\_{F}^{2}\right)$ 依赖模型参数 $\boldsymbol{W}^{(1)}$ 和 $\boldsymbol{W}^{(1)}$ 的当前值，而这些当前值是优化算法最近⼀次根据反向传播算出梯度后迭代得到的。

另⼀⽅⾯，反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传播计算得到的。举例来说，参数梯度 $\partial J/\partial \boldsymbol{W}^{(2)}=(\partial J / \partial \boldsymbol{o}) \boldsymbol{h}^{\top}+\lambda \boldsymbol{W}^{(2)}$ 的计算需要依赖隐藏层变量的当前值 $h$。这个当前值是通过从输⼊层到输出层的正向传播计算并存储得到的。

因此，在模型参数初始化完成后，我们交替地进⾏正向传播和反向传播，并根据反向传播计算的梯度迭代模型参数。既然我们在反向传播中使⽤了正向传播中计算得到的中间变量来避免重复计算，那么这个复⽤也导致正向传播结束后不能⽴即释放中间变量内存，这是训练要⽐预测占⽤ 更多内存的⼀个重要原因。另外需要指出的是，这些中间变量的个数⼤体上与⽹络层数线性相关， 每个变量的⼤小跟批量⼤小和输⼊个数也是线性相关的，它们是导致较深的神经⽹络使⽤较⼤批量训练时更容易超内存的主要原因。

## 参考链接

「动手学深度学习」正向传播、反向传播和计算图  [https://zh.d2l.ai/chapter_deep-learning-basics/backprop.html](https://zh.d2l.ai/chapter_deep-learning-basics/backprop.html)
