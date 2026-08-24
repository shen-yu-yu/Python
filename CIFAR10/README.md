# CIFAR-10 CNN 图像分类

使用 PyTorch 在 CIFAR-10 数据集上训练卷积神经网络，完成 10 类图像分类。

## 环境要求

- Python 3.8+
- 可选：NVIDIA GPU + CUDA（训练会明显更快）

## 安装

```bash
# 创建并激活虚拟环境（推荐）
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

安装 PyTorch（按你的环境二选一）：

```bash
# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版本（示例：CUDA 12.4，其他版本见 https://pytorch.org）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

安装其余依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
CIFAR10/
├── config.py      # 超参数与路径配置
├── data.py        # 数据下载、划分与 DataLoader
├── model.py       # CIFAR10CNN 模型定义
├── train.py       # 训练脚本
├── test.py        # 加载 checkpoint 与数据
├── predict.py     # 可视化测试集样本
├── requirements.txt
├── cifar10/       # 数据集（自动下载，已 gitignore）
├── checkpoints/   # 模型权重（训练后生成）
└── runs/          # TensorBoard 日志（训练后生成）
```

## 使用方法

### 训练

```bash
python train.py
```

首次运行会自动下载 CIFAR-10 到 `cifar10/`。训练结束后权重保存到 `checkpoints/model.pth`。

训练过程会写入 TensorBoard 日志：

```bash
tensorboard --logdir runs
```

在浏览器打开 http://localhost:6006 查看 loss 与验证准确率曲线。

### 预测可视化

加载 checkpoint 并显示一张测试集图片：

```bash
python predict.py
```

### 检查数据加载

```bash
python data.py
```

会打印 train / val / test 的 batch 数量与样本形状。

### 模型结构自检

```bash
python model.py
```

## 配置说明

在 `config.py` 中可修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 128 | 批大小 |
| `NUM_EPOCHES` | 15 | 训练轮数 |
| `LEARNING_RATE` | 1e3 | Adam 学习率 |
| `VAL_RATIO` | 0.1 | 从训练集划出的验证集比例 |
| `NUM_WORKERS` | 0 | DataLoader 工作进程数 |
| `DEVICE` | 自动 | 有 CUDA 时用 GPU，否则 CPU |

## 模型结构

`CIFAR10CNN` 为三层卷积 + 两层全连接：

- 输入：32×32 RGB 图像
- 卷积块：32 → 64 → 128 通道，每层后接 ReLU 与 MaxPool
- 分类头：Flatten → Linear(2048, 256) → Dropout(0.5) → Linear(256, 10)
- 训练时使用 RandomCrop、RandomHorizontalFlip 与归一化；验证/测试仅归一化

## 数据集

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 包含 10 个类别，每类 6000 张 32×32 彩色图：

- 训练集：50,000（其中 10% 划为验证集）
- 测试集：10,000

类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 许可证

本项目仅供学习使用。CIFAR-10 数据集请遵循其原始许可与引用要求。
