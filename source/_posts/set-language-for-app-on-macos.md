---
title: macOS App 语言设置
date: 2019-09-26
tag:
- mac tips
---

自 macOS High Sierra（10.13）开始，许多内置应用都有了中文翻译，例如：Finder -> 访达、Safari -> Safari 浏览器、LaunchPad -> 启动台等等。相信有很多人和我一样，不是很喜欢这种翻译，但是如果将系统语言切换为英文，那么许多其他软件也都变为英文，使用上不那么便利，这时，我们可以通过如下命令为应用单独设置语言：

```bash
defaults write $(mdls -name kMDItemCFBundleIdentifier -raw /Applications/Microsoft\ Word.app) AppleLanguages '("zh-Hans")'
```

其中，`mdls -name kMDItemCFBundleIdentifier -raw /Applications/Microsoft\ Word.app` 命令是为了获取到应用的标志。
