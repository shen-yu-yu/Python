# PyTorch 学习与实践完全指南

> 面向初学者的实战路线图：从「会用」到「读懂别人的代码」再到「自己写项目」  
> 生成日期：2026-08-24

---

## 目录

1. [学习 PyTorch 最快的方法](#1-学习-pytorch-最快的方法)
2. [函数在哪个模块？速查表](#2-函数在哪个模块速查表)
3. [读懂别人代码：推荐清单](#3-读懂别人代码推荐清单)
4. [五步读代码法（通用）](#4-五步读代码法通用)
5. [四周学习计划](#5-四周学习计划)
6. [训练循环模板（必背）](#6-训练循环模板必背)
7. [常见报错与排查](#7-常见报错与排查)
8. [每天怎么学（1～2 小时）](#8-每天怎么学12-小时)
9. [资源链接汇总](#9-资源链接汇总)

---

## 1. 学习 PyTorch 最快的方法

### 核心原则

| 原则 | 说明 |
|------|------|
| **先跑通，再理解** | 先让代码能训练、能出结果，再回头搞懂每一行 |
| **一个项目贯穿** | 用 CIFAR-10 分类器走通全流程，比看 10 个零散教程更有效 |
| **只学当前用到的 20%** | 图像分类只需：Tensor、Dataset/DataLoader、nn.Module、loss、optimizer、训练循环 |
| **每天写代码** | 看 1 小时不如自己写 30 分钟、改 30 分钟 |

### 记忆口诀

```
torch        → 算、练、存（张量、网络、训练、保存）
torchvision  → 看、用（数据集、预训练模型）
transforms   → 变（把图片变成模型能吃的 Tensor）
```

> 注意：正确模块名是 `torchvision.transforms`（复数），不是 `transformer`。

### 最小知识集（够做分类/检测入门）

```
torch
├── Tensor 操作、device、GPU
├── nn.Module（自定义网络）
├── nn / nn.functional（层与激活）
├── optim（Adam、SGD）
├── utils.data（Dataset、DataLoader）
└── save / load（模型持久化）

torchvision
├── datasets（CIFAR10、MNIST、ImageFolder）
├── transforms（预处理、数据增强）
└── models（ResNet 等预训练，可选）
```

**暂时不用深啃**：分布式训练、自定义 autograd、C++ 扩展、TorchScript 编译。

---

## 2. 函数在哪个模块？速查表

### 按「你想做什么」查找

| 你想做的事 | 导入方式 | 示例 |
|------------|----------|------|
| 创建张量、搬到 GPU | `import torch` | `torch.randn(2,3,32,32).to(device)` |
| 定义卷积/全连接层 | `import torch.nn as nn` | `nn.Conv2d(3, 64, 3)` |
| 损失函数 | `torch.nn` | `nn.CrossEntropyLoss()` |
| 优化器 | `import torch.optim as optim` | `optim.Adam(model.parameters(), lr=1e-3)` |
| 数据加载 | `torch.utils.data` | `DataLoader(dataset, batch_size=64)` |
| 下载 CIFAR-10 | `torchvision.datasets` | `datasets.CIFAR10(root=..., download=True)` |
| 图片转 Tensor、增强 | `torchvision.transforms` | `transforms.ToTensor()`、`RandomHorizontalFlip()` |
| 预训练 ResNet | `torchvision.models` | `models.resnet18(weights=...)` |
| 保存/加载模型 | `torch` | `torch.save(state_dict, path)` |

### 典型 import 模板（CIFAR-10 / 图像分类）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models   # 可选：预训练
```

### 三个实用查找技巧

**① IDE 跳转（最快）**  
Ctrl + 左键点击函数名 → 跳到定义，立刻知道来源。

**② Python 终端查模块**

```python
import torch.nn as nn
print(nn.Conv2d.__module__)   # torch.nn.modules.conv
help(nn.CrossEntropyLoss)     # 完整文档
```

**③ 看已有代码的 import**  
别人怎么 import，你就从哪用。例如 `DataLoader` 来自 `torch.utils.data`，属于 `torch` 包，不是 `torchvision`。

---

## 3. 读懂别人代码：推荐清单

> 原则：**从短、干净、能跑的项目开始**，不要一上来就啃 YOLO 整个仓库。

### 难度递进表

| 阶段 | 读什么 | 规模 | 目标 |
|------|--------|------|------|
| ★☆☆☆☆ | PyTorch 官方 MNIST 快速入门 | ~100 行 | 认识标准训练结构 |
| ★☆☆☆☆ | 你自己的 DigitVision | 小项目 | 对照理解配置、engine、model 分层 |
| ★★☆☆☆ | PyTorch 官方 CIFAR-10 教程 | ~150 行 | 彩色图 + 简单 CNN |
| ★★☆☆☆ | github.com/pytorch/examples/mnist | 单文件 | 最简可运行范例 |
| ★★★☆☆ | torchvision/models/resnet.py | 单模型文件 | 学 nn.Module 嵌套与 forward |
| ★★★☆☆ | karpathy/minGPT 或 nanoGPT | 小仓库 | 学清晰的项目组织 |
| ★★★★☆ | Ultralytics YOLO（按文件读） | 大仓库 | 学工业级训练管线 |

---

### 3.1 PyTorch 官方教程（首选）

| 教程 | 链接 |
|------|------|
| 60 分钟入门 | https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html |
| MNIST 快速入门 | https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html |
| CIFAR-10 分类 | https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html |
| 数据加载专题 | https://pytorch.org/tutorials/beginner/basics/data_tutorial.html |
| 自定义 nn.Module | https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html |

**读法**：先全局搜索 `for epoch`，定位训练循环，再往上追数据、往下追 `model`。

---

### 3.2 你自己的 DigitVision（强烈推荐第二份读）

路径：`D:\code\py\DigitVision`

```
DigitVision/
├── train.py              # 主流程：配置 → 数据 → 训练 → 保存
├── test.py               # 推理/测试入口
├── configs/config.yaml   # 超参数与路径
├── model/
│   └── digit_classifier.py   # 网络结构（nn.Module）
└── utils/
    ├── config.py         # YAML 加载
    └── engine.py         # train_step / eval_step
```

**推荐阅读顺序**：`config.yaml` → `train.py`（主循环）→ `digit_classifier.py` → `engine.py`

**对照问题**（读的时候问自己）：
- batch 从哪来？shape 是多少？
- loss 用的什么？label 是什么类型？
- 模型保存在哪一行？

---

### 3.3 pytorch/examples（GitHub）

仓库：https://github.com/pytorch/examples

| 目录 | 内容 | 建议 |
|------|------|------|
| `mnist/` | 最简完整训练 | **第一个 clone 读的** |
| `dcgan/` | 生成对抗网络 | 学完分类后再看 |
| `fast_neural_style/` | 风格迁移 | 看项目结构 |
| `imagenet/` | ImageNet 训练 | 进阶，先跳过 |

---

### 3.4 torchvision 源码（学标准写法）

在 Python 里定位文件：

```python
from torchvision.models import resnet18
import inspect
print(inspect.getfile(resnet18))
# 通常在 site-packages/torchvision/models/resnet.py
```

**重点阅读**：
1. `class ResNet(nn.Module)` 的 `__init__`：层怎么堆
2. `def forward(self, x)`：张量 shape 怎么变
3. `BasicBlock` / `Bottleneck`：子模块怎么组合

**不必一次读完**，先搞懂 `forward` 一条路径即可。

---

### 3.5 小型优质开源项目

| 项目 | 地址 | 适合学什么 |
|------|------|------------|
| minGPT | https://github.com/karpathy/minGPT | 极简、可读性极高的模型代码 |
| nanoGPT | https://github.com/karpathy/nanoGPT | 单文件训练脚本怎么写 |
| timm | https://github.com/huggingface/pytorch-image-models | 工业级 CNN，选一个 resnet 读 |
| Pytorch-UNet | https://github.com/milesial/Pytorch-UNet | 图像分割，比分类多一步 |

---

### 3.6 Ultralytics YOLO（你已在用，按顺序读）

仓库：`D:\code\yolo\ultralytics-main`

**不要从根目录乱翻**，按此顺序：

```
第 1 步  ultralytics/cfg/models/26/yolo26.yaml   # 网络结构（YAML 配置）
第 2 步  ultralytics/nn/tasks.py                 # 配置如何变成 nn.Module
第 3 步  ultralytics/data/dataset.py             # 图片和标签怎么读
第 4 步  ultralytics/engine/trainer.py           # 训练主循环（较长，分段读）
第 5 步  你自己的训练命令 / train 脚本入口
```

**读之前先回答三个问题**：
1. 数据从哪进？（dataset → dataloader）
2. loss 在哪算？
3. `optimizer.step()` 在哪？

---

### 3.7 不建议一开始读的

- 整个 Detectron2、MMDetection（体量太大）
- PyTorch C++ 源码 `torch/csrc`（初学不必）
- 论文与代码对不上的复现仓库
- 没有 README、没有可跑入口的零散脚本

---

## 4. 五步读代码法（通用）

适用于任何 PyTorch 项目（包括你自己的和开源的）。

### 第一步：画三条线（全局搜索）

在项目中搜索：

```
DataLoader    或  train_loader
model(        或  outputs = model
loss.backward()
```

连起来就是：**数据 → 前向 → 损失 → 反向 → 更新**。

### 第二步：只读「快乐路径」

假设输入正常、不报错，主流程走哪条分支？  
先忽略：`if debug`、`try/except`、日志、分布式、多 GPU。

### 第三步：用「入口 → 出口」读每个函数

```python
def train_one_epoch(loader, model, criterion, optimizer, device):
    """
    输入：loader, model, criterion, optimizer, device
    输出：该 epoch 平均 loss
  中间：for batch → to(device) → forward → loss → zero_grad → backward → step
    """
```

每个函数先写一句「干什么」，再进细节。

### 第四步：用假数据打印 shape

```python
import torch

model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 32, 32)   # CIFAR-10: NCHW
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)       # 分类一般是 [1, num_classes]
```

比干看 `forward` 快 10 倍。

### 第五步：改一行、跑一下

| 改动 | 观察什么 |
|------|----------|
| `batch_size` 改大/改小 | 显存、速度 |
| `lr` 改成 0.1 | loss 是否爆炸 |
| 注释掉 `RandomHorizontalFlip` | 准确率变化 |
| 少训 1 个 epoch | 流程是否跑通 |

**能改能动，才算读懂一层。**

---

## 5. 四周学习计划

### 第 1 周：张量 + 数据

**目标**：能加载数据、看懂 shape、会搬到 GPU。

```python
import torch
x = torch.randn(2, 3, 32, 32)  # batch, channel, H, W
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print(x.shape, x.device)
```

**练习**：CIFAR-10 取一个 batch，打印 `images.shape`、`labels`。

**阅读**：官方 MNIST 快速入门 + DigitVision 的 `train.py` 数据部分。

---

### 第 2 周：模型 + 训练循环

**目标**：自己写 CNN，完整训练 10 个 epoch，验证准确率 > 60%。

**练习**：`Conv2d`、`Linear`、`ReLU`、`MaxPool2d`、`CrossEntropyLoss`、`Adam`。

**阅读**：官方 CIFAR-10 教程 + DigitVision 的 `model/` 和 `engine.py`。

---

### 第 3 周：调参 + 保存 + 推理

**目标**：数据增强、学习率调度、保存/加载、单张图预测。

**练习**：
- `RandomCrop`、`RandomHorizontalFlip`
- `StepLR` 或 `CosineAnnealingLR`
- `torch.save` / `torch.load`

**阅读**：`torchvision/models/resnet.py`（只看结构）。

---

### 第 4 周：读工业代码

**目标**：能定位 YOLO 训练入口，知道数据→loss→step 在哪。

**阅读**：Ultralytics 的 `dataset.py` + `trainer.py`（分段）。

---

### 每周检验标准

| 周 | 你能做到 |
|----|----------|
| 1 | 不看文档写出 DataLoader + ToTensor |
| 2 | 不看文档写出完整训练循环 |
| 3 | 保存模型并加载预测一张图 |
| 4 | 在 YOLO 项目里找到 trainer 的三条主线 |

---

## 6. 训练循环模板（必背）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    # ---------- 训练 ----------
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------- 验证 ----------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

# ---------- 保存 ----------
torch.save(model.state_dict(), "model.pth")
```

---

## 7. 常见报错与排查

| 报错 | 常见原因 | 解决办法 |
|------|----------|----------|
| `size mismatch` | 输入 shape 与层定义不符 | 打印每层输入输出 shape |
| `CUDA out of memory` | batch 太大或模型太大 | 减小 batch_size |
| `expected scalar type Long but found Float` | 分类 label 应是 int64 | `labels.long()` |
| `grad can be implicitly created only for scalar` | loss 不是标量就 backward | 检查 loss 是否 `.mean()` |
| `No module named torchvision` | 环境没装 | `pip install torchvision` |
| pip 装 lxml 编译失败 | 缺 C++ 编译器 | `conda install -c conda-forge lxml` |

**通用习惯**：报错先看**最后 5 行**；训练时随手 `print(tensor.shape)`。

---

## 8. 每天怎么学（1～2 小时）

| 时间 | 做什么 |
|------|--------|
| 20 分钟 | 看官方文档/教程的**一个小节** |
| 40 分钟 | **自己敲代码**（尽量不复制粘贴） |
| 20 分钟 | 改一个参数，观察 loss/准确率变化 |
| 10 分钟 | 记笔记：今天学了什么、卡在哪 |

**比多看教程更有效的事**：
- 把 `lr=0.001` 改成 `0.1`，看 loss 会不会炸
- 把 `batch_size` 从 64 改成 4，看训练速度
- 注释掉数据增强，看准确率差多少

---

## 9. 资源链接汇总

### 官方

- PyTorch 文档：https://pytorch.org/docs/stable/
- Torchvision 文档：https://pytorch.org/vision/stable/
- 教程首页：https://pytorch.org/tutorials/

### 代码仓库

- PyTorch Examples：https://github.com/pytorch/examples
- minGPT：https://github.com/karpathy/minGPT
- nanoGPT：https://github.com/karpathy/nanoGPT
- timm：https://github.com/huggingface/pytorch-image-models
- Ultralytics：https://github.com/ultralytics/ultralytics

### 本地项目

- DigitVision（MNIST）：`D:\code\py\DigitVision`
- Ultralytics YOLO：`D:\code\yolo\ultralytics-main`

---

## 附录 A：CIFAR-10 项目目标（建议的下一个实战）

**任务**：输入 32×32 彩色图，分类为 10 类（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。

**将覆盖的知识点**：
- Tensor 与 GPU
- Dataset、DataLoader
- nn.Module 与自定义 CNN
- 卷积、池化、全连接
- 损失函数与优化器
- 完整训练循环
- 验证集评估
- 模型保存与加载
- 数据增强与学习率调度

---

## 附录 B：一句话总结

> **用一个完整小项目（CIFAR-10）走通全流程；读代码从官方 MNIST → DigitVision → ResNet → YOLO trainer；每天写代码、改参数、看 shape 和报错；不会就 Ctrl+点击或 help()。**

---

*本文档由 Cursor AI 助手根据学习对话整理生成。*