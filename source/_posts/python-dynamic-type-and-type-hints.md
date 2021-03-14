---
title: Python 动态类型以及类型提示
date: 2019-11-01
tags:
  - Python
  - 动态类型
  - 类型提示
---

随着机器学习和数据分析变得越来越热门，Python 作为一门**解释型**和**动态类型**语言，很好的顺应了这一潮流，成为最流行的语言之一。 解释型语言的主要特点就是执行代码之前不需要编译，利用 [Jupyter Notebook](https://jupyter.org/) 等交互式的工具，可以方便快速的测试一些想法；而动态类型可以摆脱严格的继承关系或接口实现的束缚，简化程序的设计或实现。本文主要谈一下对动态类型的理解，以及类型提示的作用和重要性。

<escape><!-- more --></escape>

## 动态类型

### 变量的类型？

在 Python 语言中，变量不需要指定类型，甚至可以随时赋值为其它类型：

```python
a = 'hello world'
print(a)  # hello world

a = 8
print(a)  # 8
```

这是不是意味着，Python 中没有类型呢？答案显然是否定的。因为，初学 Python 时，首先，接触的就是数据类型，Python 为我们提供了布尔、数字（int，float 和 complex）、序列（list、tuple 和 range）、字符序列（str）、二进制序列（bytes、bytearray 和 memoryview）、集合（set）和字典（dict）等类型（参见 Python 文档中[内置类型]( https://docs.python.org/3/library/stdtypes.html)一节）。

```python
a = 'hello world'
print(a.split(' '))  # ['hello', 'world']

a = 8
print(a.split(' '))  # AttributeError: 'int' object has no attribute 'split'
```

如果尝试执行上面的代码，会发现第 2 行可以正常执行，而第 5 行则会发生 `AttributeError` 错误。那么该如何理解这一现象呢？我总结了下面这句话（不记得哪里看到的了 🤔）：**变量没有类型，但是数据是有类型的**，变量的行为由其指向的数据所决定。

### 鸭子 🦆 类型

所谓的鸭子类型来源于鸭子测试，即：“当看到一致鸟**走**起来像鸭子、**游泳**的时候像鸭子、**叫**起来也像鸭子，那么这只鸟就可以被称为鸭子”。可以看到，在鸭子类型中，关注点在于对象的行为，而不是对象所属的类型。用专业一点的术语来说就是：一个对象的有效语意是由其方法和属性的集合决定的，而不是继承特定的类或实现特定的接口。

下面用一个例子来说明鸭子类型：

```python
class Duck(object):
    def run(self):
        print('a duck is running')

    def swim(self):
        print('a duck is swimming')

    def quack(self):
        print('嘎嘎嘎')


class Bird(object):
    def run(self):
        print('a bird is running')

    def swim(self):
        print('a bird is swimming')

    def quack(self):
        print('唧唧唧')
```

上面定义了 `Duck` 和 `Bird` 两种类型，都继承自 `object`，互相之间没有继承关系，但是都定义了 `run`、`swim` 和 `quack` 三个方法，如果将 `Duck` 和 `Bird` 的实例传入下面的 `duckTest` 函数，程序都可以正常运行。

```python
def duckTest(duck):
    duck.run()
    duck.swim()
    duck.quack()


bird = Bird()
duckTest(bird)
>>> a bird is running
a bird is swimming
唧唧唧

duck = Duck()
duckTest(duck)
>>> a duck is running
a duck is swimming
嘎嘎嘎
```

## 类型提示

动态类型为程序设计带来了极大的便利性，同时，也向程序员提出了更高的要求，即：在编写程序时，必须很好地理解项目中的每一行代码。虽然，每个程序员应该具备这样的能力，或者说应该做到，但是，随着项目中代码量的增大，这一点会变得越来越难。相信大家在使用 Python 编程的时候都遇到过下面这两个问题：

1. **函数的参数类型**：在定义一个方法时，我们不需要**显示**指出参数的类型，但是，我们在方法中对该参数进行的所有操作实际上都是基于某一类型设计的，如果传入参数的不符合这一预期，程序将会出现错误。可以说该类型存在与函数设计者的脑子里，而使用者需要阅读文档或者查看源码才知道如何传入一个恰当的值。
2. **代码补全**：`IDE` 的一个核心功能就是代码提示，但是，由于方法定义时没有指出参数的类型，`IDE` 只能根据上下文进行一些推测，导致代码提示不准确或者说根本无法进行提示。

类型提示就是为了解决这个问题，[PEP 484](https://www.python.org/dev/peps/pep-0484/) 对类型提示进行了详细的描述。下面是一个加入类型提示的方法定义的示例：

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

代码中的类型标注可以帮助我们解决上面提到的两个问题，并且，Python 并不会在函数执行的时候对参数的类型进行校验，所以丝毫不影响原来代码的执行。此外，借助 [mypy](https://github.com/python/mypy) 或 [pyright](https://github.com/microsoft/pyright) 等静态类型分析工具还可以帮助我们发现程序中潜在的一些问题。