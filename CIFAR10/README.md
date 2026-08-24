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
├── src/                   # 源码：配置、数据、模型、工具
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── scripts/               # 入口脚本
│   ├── train.py
│   ├── test.py
│   └── predict.py
├── docs/                  # 文档
│   └── 项目问题与改进清单.md
├── train.py               # 根目录兼容入口（转发到 scripts/）
├── test.py
├── predict.py
├── data/                  # 自动下载的数据集（gitignore）
├── outputs/               # 权重与日志（gitignore）
│   ├── checkpoints/
│   └── runs/cifar10/
├── requirements.txt
└── README.md
```

首次运行会自动下载 CIFAR-10 到 `data/`。`outputs/` 目录会在训练时自动创建，无需手动新建。

训练结束会保存两个权重文件：

| 文件 | 含义 |
|------|------|
| `outputs/checkpoints/best.pth` | 验证集准确率最高时的权重 |
| `outputs/checkpoints/model.pth` | 最后一个 epoch 的权重 |

`test.py` 与 `predict.py` 默认加载 `outputs/checkpoints/best.pth`。若要用最后一个 epoch 的权重，将其中的 `CHECKPOINT_BEST` 改为 `CHECKPOINT_RESULT`。

更多踩坑记录见 [docs/项目问题与改进清单.md](docs/项目问题与改进清单.md)。

## 使用方法

以下命令在项目根目录执行即可（根目录的 `train.py` 等会自动转发到 `scripts/`）：

### 训练

```bash
python train.py
# 或
python scripts/train.py
```

默认训练 15 个 epoch，可在 `src/config.py` 中修改 `NUM_EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE` 等。

查看 TensorBoard：

```bash
tensorboard --logdir outputs/runs/cifar10
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

主要配置见 `src/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 128 | 批大小 |
| `NUM_EPOCHS` | 15 | 训练轮数 |
| `LEARNING_RATE` | 1e-3 | Adam 学习率 |
| `VAL_RATIO` | 0.1 | 从训练集划出的验证集比例 |
| `DEVICE` | 自动 | 有 CUDA 用 GPU，否则 CPU |

## 类别

CIFAR-10 共 10 类：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck。
