---
title: 深入理解 Vue 响应式系统
date: 2020-04-06
tags:
  - Vue
  - 响应式
  - MVVM
---

Vue 是目前主流的前端框架之一，相信也是其中最容易上手的（没有之一），开发者只需具备 HTML、CSS 和 JavaScript 等基础知识，就可以根据[官方教程](https://cn.vuejs.org/v2/guide/)开启 Vue 编程之旅。Vue 推崇使用 `template` 的方式实现页面，这使得开发者很容易将设计好的原型改造成 Vue 单文件组件，利用 Vue 提供的响应式系统，我们只需要关注**数据**的存储和变化，Vue 会自动在数据发生变化的时候更新页面的内容。那么这一切是如何实现的呢？本文将结合 MVVM（Model-View-ViewModel） 设计模式，剖析 Vue 响应式系统的原理。

<escape><!-- more --></escape>

## MVVM

MVVM 模型由 Model、View 和 ViewModel 三部分组成，简单来讲，Model 是我们定义的数据和这些数据的业务逻辑；View 是这些数据展示给用户的形式；而 ViewModel 则是连接 Model 和 View 的桥梁，通过数据绑定保证了 Model 和 View 的一致性，即：当 Model 变化时，自动更新 View；当用户操作 View，并作出了更改时，同步 View 的数据到 Model。

{% asset_img mvvm.png 600 [MVVM 模型] %}

从上图可见，VM 是 MVVM 模型的关键，而数据绑定又 VM 的核心技术。通常， ModelView 由框架来实现，Model 和 View 则由开发者自己定义。

## Vue 模版组件示例

下面我们通过一个 Vue 模版组件示例来体会一下 MVVM 模型为开发者带来的好处：

> 可以在 [CodeSandbox](https://codesandbox.io/) 中运行 https://codesandbox.io/s/vue-template-component-0692e

```vue
<template>
  <div>
    <h1>Greeting</h1>
    <div>
      <label>username:</label>
      <input v-model="username">
    </div>
    <span>hello, {{ username }}!</span>
  </div>
</template>

<script>
  export default {
    name: "Greeting",
    data() {
      return {
        username: "world",
      };
    },
  };
</script>
```

如果你对 Vue 有一定的了解，相信不用运行也能想象出这段代码的执行效果。这个组件的内容展示和交互都非常简单，只有一个输入框供用户输入数据，但是，当输入框中的输入数据变化，下面的文字也会跟着变化。结合上面讲到的 MVVM 模型，我们可以很容易的将组件中的 `template` 和 `script` 模块和 MVVM 模型中的 View 和 Model 对应起来。那么 VM 是谁来实现的呢？答案就是 Vue。

在阅读 Vue 官方文档的时候，我们经常会看到类似下面的代码：

```javascript
var vm = new Vue({
  // 选项
})
```

按照[官方的说法]([https://cn.vuejs.org/v2/guide/instance.html#%E5%88%9B%E5%BB%BA%E4%B8%80%E4%B8%AA-Vue-%E5%AE%9E%E4%BE%8B](https://cn.vuejs.org/v2/guide/instance.html#创建一个-Vue-实例))：「Vue **没有完全遵循** MVVM 模型，但是 Vue 的设计也受到了它的启发。因此在文档中经常会使用 `vm`（ViewModel 的缩写）这个变量名表示 Vue 实例 」。

> 注意上面加粗的几个字，我们在后面会做进一步分析。

## Vue 响应式系统

关于 Vue 的响应式系统，Vue 官方教程中有一节专门的[描述](https://cn.vuejs.org/v2/guide/reactivity.html)。下图就来自该文档，从这幅图中，我们可以对 Vue 响应式系统有一个大致的了解。

{% asset_img vue_reactivity.png 600 [Vue 响应式实现] %}

总体上来说，要实现一个响应式系统，至少需要完成两件事：1. **变化侦测**：感知数据的变化；2. **依赖收集**：收集哪些地方使用了数据，以便在数据发生变化时通知其进行更新。

### 变化侦测

对于对象和数组，Vue 采用了不同的方式来观测其变化，下面是 Vue 2.x 及之前版本的实现方式。

#### 对于对象

Vue 采用的是 `Object.defineProperty()` 方法来实现的，即：对组件中 `data` 函数返回对象中的每一个属性通过 `Object.defineProperty()` 转换为 `getter/setter` 。通过这种方式，Vue 就可以感知到程序运行中对变量的赋值操作。

> Vue 源码：https://github.com/vuejs/vue/blob/dev/src/core/observer/index.js

#### 对于数组
数组与对象不同，Vue 通过覆盖 Array 原型方法的方式来实现的。JavaScript 中改变数组元素的方法有：`push`、`pop`、`shift`、`unshift`、`splice`、`sort` 和`reverse` 共 7 个，只要拦截 Array 的这些操作我们就可一获知绝大多数情况下数组中元素的变化。

>  Vue 源码：https://github.com/vuejs/vue/blob/dev/src/core/observer/array.js

#### 注意事项

由于实现方式的限制，Vue 无法侦测一些变化，如果对这些操作缺乏了解，可能导致项目中出现一些预期之外的 Bug，即：修改了数据，但是却没有触发页面更新或者对应的 watch 方法。

1. 无法检测属性的添加和删除；
2. 对于已经创建的实例，无法添加根级别的响应式属性；
3. 利用缩影直接设置一个数组项的值；
4. 修改数组长度

### 依赖收集

侦测到数据变化后，我们还需要更新使用了这个数据的地方，如：计算属性、`watch` 函数或者 `template` 模版，如何判断一个变量或属性在哪些地方使用了呢？这就轮到 `getter` 发挥作用了。我们可以认为使用了某个变量的地方依赖了这个变量，而使用的时候必定会触发其 `getter` 函数，所以我们可以在 `getter` 函数中进行相关的依赖收集，并在数据变化的时候通知这些依赖。

结合上面的 Vue 响应式原理图，我们所写的 `template` 最终会被 [vue-template-compiler](https://github.com/vuejs/vue/tree/dev/packages/vue-template-compiler) 转换为 render 函数，再通过 render 函数生成虚拟 DOM 树，在生成虚拟 DOM 树的时候，因为我们使用了 `v-model` 和 “Mustache” 语法 (双大括号) 绑定了 `username`，所以，Vue 将 render 函数视为 `username` 的一个依赖，当我们在输入框中输入文字时，由于使用了双向绑定，`username` 会更新为当前输入的值，进一步触发 `re-render`，`span` 中的 `username` 也更新为最终的值。

## 相关链接

- [维基百科：MVVM](https://zh.wikipedia.org/wiki/MVVM)
- [Vue 官方教程：深入响应式原理](https://cn.vuejs.org/v2/guide/reactivity.html)
- [阮一峰：MVC，MVP 和 MVVM 的图示](http://www.ruanyifeng.com/blog/2015/02/mvcmvp_mvvm.html)
- [廖雪峰：MVVM](https://www.liaoxuefeng.com/wiki/1022910821149312/1108898947791072)
- [MDN: Object.defineProperty](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Object/defineProperty)
- [MDN: 使用对象](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Guide/Working_with_Objects)
- [深入浅出 Vue.js](https://book.douban.com/subject/32581281/)