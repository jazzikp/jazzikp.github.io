---
layout:     post
title:      Python for Absolute Newbies
title_zh: "Python 零基础入门"
subtitle:   Python installation, pip, and some basic functionality
subtitle_zh: "Python 安装、pip 和基本功能"
date:       2019-04-03
bilingual: true
author:     Zhejian Peng
header-img: img/python-tutorial-for-beginners.webp
catalog: true
tags:
    - Python
    - Data Science
---

{::options parse_block_html="true" /}

<div data-lang-panel="en" markdown="1">

这篇文章会从安装 Python3.7 开始讲起。
接下来的主题会渐渐集中讲一些 Data Science 的一些应用和建模方法。
使用 macOS Mojave.

## Zen of Python

Python was designed as a successor to the [ABC language](https://en.wikipedia.org/wiki/ABC_(programming_language)) by
[Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum). 
Python was initially developed as a "hobby" project that would keep him occuiped during the Christmas break! I mean who would spend their Christmas writing an [interpreter](https://en.wikipedia.org/wiki/Interpreter_(computing))!

Eventually, Python become one of the top used programming language today. Python can be used for all kinds of projects from small personal project to large application, and from web development to scientific computation. Here are some famous projects used Python, according to this [article](https://www.hartmannsoftware.com/Blog/Articles_from_Software_Fans/Most-Famous-Software-Programs-Written-in-Python)

- YouTube
- Google search engine also used Python for its mainframe
- Instagram
- Reddit

Let's start this tutorial with the Zen of Python:

Type this into the Mac terminal.

```bash
python
>>> import this
```

## Install Python3.7

Let's see how to install Python3.6. Mac come with default python2.7 installed. We want to use Python3.7 here.

I found the easiest way here is to install through [Anaconda Distribution](https://repo.anaconda.com/archive/Anaconda3-2018.12-MacOSX-x86_64.pkg) (Click this link to download directly!)

You can also install through [python.org](https://www.python.org/). 

Please also install VS Code come with Anaconda. I will talk about Visual Studio Code later.
If you installed through python.org, please also install Visual Studio Code separately.

Once you have Python3.7 installed, you should be able to see this in terminal
```bash
~ » python3.7
Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24)
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```

Congrats! You know have the latest version of Python ready!

## Virtual Environment

Do use a virtual environment, PLEASE!!! Virtual environment allow you to separate your developing environment without interfering with the system. 

I want you to have the best programming habit from the beginning. I learned this the hard way.

Lets try to setup virtual environment using virtualenv.

First go to home directory and create a directory Environment to store all your virtual environments
```bash
~ » cd ~
~ » mkdir Environment
~ » cd Environment
```

Then install virtualenv and create a new virtual environment named it python_tutorial

```bash
~ » pip install virtualenv
~ » virtualenv ~/Environment/python_tutorial
```

Activate python_tutorial. Now type which python3. It will return the path to the virtual environment.

```bash
~ » source ~/Environment/python_tutorial/bin/activate
~ » which python3
/Users/zhejianpeng/Environment/python_tutorial/bin/python3
```

Deactivate python_tutorial environment

```bash
~ » deactivate
```

Now every time before you start coding, remember to activate the virtual environment. It's recommended to create a short alias of the activation command and add it to ~/.bash_profile. You don't have to type the long version.

```bash
~ » code ~/.bash_profile
# User Define Alias, add following line to your bash_profile
alias tutorial='source ~/Environment/python_tutorial/bin/activate'
``` 

Here you need to install code command. It's very easy. Open VS code you installed previously, and type <kbd>shift</kbd> + <kbd>command</kbd> + <kbd>p</kbd>. Then type ![The VS Code command palette](/img/vscode-command-palette.webp){: loading="lazy" decoding="async" width="820" height="116"}

Now you have the basic environment setup!!! Let's start coding

## Hello World

</div>

<div data-lang-panel="zh" hidden markdown="1">

这篇文章会从安装 Python3.7 开始讲起。
接下来的主题会渐渐集中讲一些 Data Science 的一些应用和建模方法。
使用 macOS Mojave.

## Python 之禅

Python 被设计为 [ABC 语言](https://en.wikipedia.org/wiki/ABC_(programming_language)) 的继任者，由 [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) 开发。
Python 最初是作为一个“业余”项目开发的，好让他在圣诞假期有事可做！谁会在圣诞节写一个 [解释器](https://en.wikipedia.org/wiki/Interpreter_(computing)) 啊！

最终，Python 成为当今使用最广泛的编程语言之一。Python 可用于各种项目，从小个人项目到大型应用，从 Web 开发到科学计算。根据这篇 [文章](https://www.hartmannsoftware.com/Blog/Articles_from_Software_Fans/Most-Famous-Software-Programs-Written-in-Python)，以下是一些使用 Python 的著名项目：

- YouTube
- Google 搜索引擎的主机也使用了 Python
- Instagram
- Reddit

我们从 Python 之禅开始这个教程：

在 Mac 终端中输入以下内容。

```bash
python
>>> import this
```

## 安装 Python3.7

看看如何安装 Python3.6。Mac 默认安装了 python2.7。我们这里要用 Python3.7。

我发现最简单的方法是通过 [Anaconda Distribution](https://repo.anaconda.com/archive/Anaconda3-2018.12-MacOSX-x86_64.pkg) 安装（点击此链接直接下载！）

也可以通过 [python.org](https://www.python.org/) 安装。

请同时安装 Anaconda 自带的 VS Code。我稍后会讲 Visual Studio Code。
如果通过 python.org 安装，请单独安装 Visual Studio Code。

安装好 Python3.7 后，终端里应该能看到这个
```bash
~ » python3.7
Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24)
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```

恭喜！你现在已经准备好最新版本的 Python 了！

## 虚拟环境

一定要用虚拟环境，拜托！！！虚拟环境可以隔离你的开发环境，不会干扰系统。

我希望你从一开始就养成最好的编程习惯。这是我吃过亏才学到的。

我们用 virtualenv 来设置虚拟环境。

首先进入主目录，创建一个 Environment 目录来存放所有虚拟环境
```bash
~ » cd ~
~ » mkdir Environment
~ » cd Environment
```

然后安装 virtualenv 并创建一个名为 python_tutorial 的新虚拟环境

```bash
~ » pip install virtualenv
~ » virtualenv ~/Environment/python_tutorial
```

激活 python_tutorial。现在输入 which python3。它会返回虚拟环境的路径。

```bash
~ » source ~/Environment/python_tutorial/bin/activate
~ » which python3
/Users/zhejianpeng/Environment/python_tutorial/bin/python3
```

停用 python_tutorial 环境

```bash
~ » deactivate
```

现在每次开始编码前，记得激活虚拟环境。建议为激活命令创建一个短别名，并添加到 ~/.bash_profile。这样就不用输入长命令了。

```bash
~ » code ~/.bash_profile
# User Define Alias, add following line to your bash_profile
alias tutorial='source ~/Environment/python_tutorial/bin/activate'
``` 

这里需要安装 code 命令。很简单。打开之前安装的 VS Code，输入 <kbd>shift</kbd> + <kbd>command</kbd> + <kbd>p</kbd>。然后输入 ![VS Code 命令面板](/img/vscode-command-palette.webp){: loading="lazy" decoding="async" width="820" height="116"}

现在基本环境已经设置好了！！！开始编码吧

## Hello World

</div>
