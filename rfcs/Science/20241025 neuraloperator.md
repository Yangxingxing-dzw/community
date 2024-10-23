

# 飞桨适配 nueralop

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |         qikai           |
| 提交时间      |       2024-10-25   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop/release 3.0.0 版本        |
| 文件名        | 2024_10_25_nueralop.md             |

## 1. 概述

### 1.1 相关背景

> 背景相关 issue 链接或者直接文字说明。

[飞桨科学计算工具组件开发大赛](https://github.com/PaddlePaddle/PaddleScience/issues/1000)

`neuraloperator`是一个用于在 PyTorch 中学习神经运算符的综合库。它是 Fourier Neural Operators 和 Tensorized Neural Operators 的官方实现。并且`neuraloperator`支持在函数空间之间学习映射。

为`neuraloperator`工具组件支持飞桨后端，可以提高飞桨用户开发科学计算模型的开发效率。

### 1.2 功能目标

> 具体需要实现的功能点。

1. 整理 `neuraloperator` 的所有公开 API；
2. 使用 paddle 的 python API 等价组合实现上述公开 API 的功能；
3. 参考 pytorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过。

### 1.3 意义

> 简述本 RFC 所实现的内容的实际价值/意义。

为`neuraloperator`支持飞桨后端，从而提高飞桨用户开发科学计算模型的开发效率。

## 2. PaddleScience 现状

> 说明 PaddleScience 套件与本设计方案相关的现状。

当前的PaddleScience有`neuraloperator`相关的[神经算子](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/neuraloperator)实现。

本方案不考虑原有PaddleScience有`neuraloperator`的实现，而是单独按照
`neuraloperator`的源码来撰写支持飞桨后端的文件。

## 3. 目标调研

> 如果是论文复现任务，则需说明复现论文解决的问题、所提出的方法以、复现目标以及可能存在的难点；如果是 API 开发类任务，则需说明已有开源代码中类似功能的实现方法，并总结优缺点。

参考的源代码为：[https://github.com/neuraloperator/neuraloperator/tree/0.3.0](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)

`neuraloperator`源代码实现的10个模型分别为`TFNO`, `TFNO1d`, `TFNO2d`, `TFNO3d`,`FNO`, `FNO1d`, `FNO2d`, `FNO3d`, `SFNO`, `UNO`。

为`neuraloperator`支持飞桨后端，主要的难点在于：

（1）傅里叶卷积层中复数权重的转换。

（2）`TFNO`中使用tltorch进行张量分解，需要利用paddle 的 python API组合实现张量分解的功能。
## 4. 设计思路与实现方案

> 结合 PaddleScience 套件的代码结构现状，描述如何逐步完成论文复现/API开发任务，并给出必要的代码辅助说明。

比赛要求：所有文件组织结构必须与原有代码保持一致（新增文件除外），原有的注释、换行、空格、开源协议等内容不能随意变动（新增内容除外），否则会严重影响代码合入和比赛结束后成果代码的维护。


所以本项目按照[`neuraloperator`](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)的源码来撰写支持飞桨后端的文件，并且保持文件结构、名称、注释、空格等都相同。

### 4.1 补充说明[可选]

> 可以是对设计方案中的名词、概念等内容的详细解释。

## 5. 测试和验收的考量

> 说明如何对复现代码/开发API进行测试，以及验收标准，保障任务完成质量。

1.模型前向对齐

1.1权重转化

为了保证模型前向对齐不受到模型参数不一致的影响，我们使用相同的权重参数对模型进行初始化。

生成相同权重参数主要分为以下 3 步：

&nbsp;&nbsp;（1）随机neuraloperator-pytorch的官方模型参数并保存成 pytorch_model.pth；

&nbsp;&nbsp;（2）将 pytorch_model.pth通过paddle官方文档中的torch2paddle.py 生成paddle_model.pdparams；

&nbsp;&nbsp;（3）将paddle_model.pdparams载入neuraloperator-paddle模型。


转换模型时，torch 和 paddle 存在参数需要转换的部分，主要是傅里叶卷积层中复数的权重的转换过程是将实部和虚部分开保存和加载。

1.2模型前向对齐验证

以FNO2d模型的前向对齐验证为例，主要分为以下4步：

&nbsp;&nbsp;（1）首先将arcy_train_16.pt、darcy_test_16.pt、darcy_test_32.pt等数据集（[https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/data](https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/datad)）转化为ndarray格式；

&nbsp;&nbsp;（2）然后分别在paddle和torch下对模型的dataloder，datasets进行了对齐，保证两者相同；

&nbsp;&nbsp;（3）PyTorch 前向传播：定义 PyTorch 模型，加载权重，固定 seed，基于 numpy 生成`channel=3`的随机数，转换为 `torch.Tensor`，送入网络，获取输出`y_pytorch`；

&nbsp;&nbsp;（4）飞桨前向传播：定义飞桨模型，加载步骤 1.1 中转换后的权重，将步骤（3）中的生成的随机数，转换为 `paddle.Tensor`，送入网络，获取输出`y_paddle`。

定义`y_diff=|y_pytorch-y_paddle|`。最终差异结果如下：

| 模型   | Max(y_diff)  | Min(y_diff)   | MSE(y_pytorch, y_paddle)   |
|:-----:|:-----:|:-----:|:-----:|
| FNO2d | 2.69e-05 | 0.00 | 3.44e-11 |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |



2.模型训练对齐：

以FNO2d模型的训练对齐为例：

FNO2d-torch的每个epoch训练loss（L2 norm）为`loss_pytorch`，学习率为`lr_pytorch`。

FNO2d-paddle的训练loss（L2 norm）为`loss_paddle`. 学习率为`lr_paddle`。

定义指标`loss_diff=|loss_pytorch-loss_paddle|`，`lr_diff=|lr_pytorch-lr_paddle|`。

FNO2d-torch与FNO2d-paddle使用数据集 darcy_train_16.npy并且设置`ntrains`=32 ，

FNO2d-torch的优化器参数为：
```python
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

```

FNO2d-paddle的优化器参数为：
```python
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.008, T_max=30)

# 创建 Adam 优化器，并将调度器作为学习率传递给优化器
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), 
                                  learning_rate=scheduler, 
                                  weight_decay=0.0001)
```


FNO2d-torch与FNO2d-paddle的训练loss和学习率对比结果如下：


| epoch   | loss_pytorch   | loss_paddle   |loss_diff|lr_pytorch|lr_paddle|lr_diff|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 | 32.06811523 | 32.06811523  |0.00|0.00797809|0.00797809|0.00|
| 1 | 31.38529968 | 31.38491058  |3.89e-04|0.00791259|0.00797809|0.00|
| 2 | 27.89798164 | 27.89617157 |1.81e-03|0.00780423|0.00797809|0.00|
| 3| 30.61170769 | 30.61891174 |7.20e-03|0.00765418|0.00797809|0.00|
| 4| 21.70384979 | 21.7020092 |1.84e-03|0.0074641|0.0074641|0.00|
| 5| 21.37845802| 21.37961197 |1.15e-03|0.00723607|0.00723607|0.00|
| 6| 20.06527138| 20.0691185|3.85e-03|0.00697258|0.00697258|0.00|
| 7| 16.89506531 | 16.89888763 |3.82e-03|0.00667652|0.00667652|0.00|
| 8| 15.51529121 | 15.51611137 |8.20e-04|0.00635114|0.00635114|0.00|
| 9| 14.76096916 | 14.75966835 |1.30e-03|0.00600000|0.00600000|0.00|



各个模型最终训练loss的差异为：
| 模型   | Max(loss_diff)  | Min(loss_diff)   | MSE(loss_pytorch, loss_paddle)   | lr_diff   |
|:-----:|:-----:|:-----:|:-----:|-----:|
| FNO2d | 7.20e-03 | 0.00 | 9.18e-06 |&nbsp; |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |&nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |&nbsp; |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |&nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |&nbsp; |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |&nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |&nbsp; |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |&nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |&nbsp; |
| &nbsp; | &nbsp; | &nbsp;  | &nbsp; |&nbsp; |
| &nbsp; | &nbsp; |&nbsp;   | &nbsp;  |&nbsp; |

使用`timeit`中的`default_timer`进行进行计时，各个模型10个epochs训练耗时`10_epochs_time`, 数据读取耗时`data_reader_time`对比结果如下：

| 模型   | 10_epochs_time_torch | data_reader_time_torch|10_epochs_time_paddle|data_reader_time_paddle| 
|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO2d | 1.74e-01 | 7.17e-03 |4.56e-01|1.84e-01| 
| &nbsp; | &nbsp; | &nbsp;  |||
| &nbsp; | &nbsp; |&nbsp;   ||| 
| &nbsp; | &nbsp; | &nbsp;  ||| 
| &nbsp; | &nbsp; |&nbsp;   ||| 
| &nbsp; | &nbsp; | &nbsp;  ||| 
| &nbsp; | &nbsp; |&nbsp;   ||| 
| &nbsp; | &nbsp; | &nbsp;  ||| 
| &nbsp; | &nbsp; |&nbsp;   ||| 
| &nbsp; | &nbsp; | &nbsp;  ||| 
| &nbsp; | &nbsp; |&nbsp;   ||| 

## 6. 可行性分析和排期规划

> 可以以里程碑列表的方式，细化开发过程，并给出排期规划。

| 里程碑        |  时间点     |
| -------------| ------------ | 
| 提交RFC      |     2024.10.25        |    
| 提交PR，修改代码完成合入          |    2024.11.25       | 
## 7. 影响面

> 描述本方案对 PaddleScience 可能产生的影响。

为 PaddleScience 添加`neuraloperator`库。
