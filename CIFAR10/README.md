# CIFAR-10 CNN 分类

使用 PyTorch 在 CIFAR-10 数据集上训练一个简单 CNN，并完成测试与单张图片预测可视化。

## 环境要求

- Python 3.10+
- 建议使用虚拟环境

## 安装

1. 创建并激活虚拟环境（可选）：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

2. 安装 PyTorch 与 torchvision（按你的 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择安装命令；无 GPU 可选 CPU 版本）。

3. 安装其余依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
CIFAR10/
├── config.py       # 超参数与路径配置
├── data.py         # 数据下载、增强、划分 train/val/test
├── model.py        # CIFAR10CNN 模型定义
├── train.py        # 训练并保存权重
├── test.py         # 在测试集上评估准确率
├── predict.py      # 可视化单张图片的预测结果
├── requirements.txt
└── README.md
```

首次运行会自动下载 CIFAR-10 到 `cifar10/`。训练日志写入 `runs/cifar10/`；`checkpoints/` 目录会在训练时自动创建，无需手动新建。

训练结束会保存两个权重文件：

| 文件 | 含义 |
|------|------|
| `checkpoints/best.pth` | 验证集准确率最高时的权重 |
| `checkpoints/model.pth` | 最后一个 epoch 的权重 |

`test.py` 与 `predict.py` 默认加载 `checkpoints/best.pth`（验证集最优权重）。若要用最后一个 epoch 的权重，将其中的 `CHECKPOINT_BEST` 改为 `CHECKPOINT_RESULT`。

## 使用方法

### 训练

```bash
python train.py
```

默认训练 15 个 epoch，可在 `config.py` 中修改 `NUM_EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE` 等。

查看 TensorBoard：

```bash
tensorboard --logdir runs/cifar10
```

### 测试

训练完成后：

```bash
python test.py
```

### 预测可视化

```bash
python predict.py
```

会弹出一张测试集图片，标题显示预测类别与真实标签。

## 配置说明

主要配置见 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 128 | 批大小 |
| `NUM_EPOCHS` | 15 | 训练轮数 |
| `LEARNING_RATE` | 1e-3 | Adam 学习率 |
| `VAL_RATIO` | 0.1 | 从训练集划出的验证集比例 |
| `DEVICE` | 自动 | 有 CUDA 用 GPU，否则 CPU |

## 类别

CIFAR-10 共 10 类：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck。
