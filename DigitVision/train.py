# ======================
# 1. 导入必要的库
# ======================

import torch  # PyTorch 主库，提供张量操作和深度学习功能
import torchvision.transforms as transforms  # torchvision 的数据预处理模块，用于图像变换
import torchvision.datasets as dataset  # torchvision 的数据集模块，提供常用数据集（如 MNIST）
import torch.utils.data as data_utils  # PyTorch 数据加载工具，用于创建 DataLoader

# 导入自定义模块
from model.digit_classifier import DigitClassifier  # 自定义的数字分类器模型类
from utils.config import load_yaml  # 自定义的 YAML 配置文件加载函数
from utils.engine import eval_step  # 自定义的评估函数（单步测试）

# ======================
# 2. 读取配置文件
# ======================

# 加载 YAML 配置文件，返回一个字典（包含训练参数、模型路径、数据路径等）
config = load_yaml("./configs/config.yaml")

# 从配置字典中提取训练参数
batch_size = config["train"]["batch_size"]  # 批大小（每次训练/测试的样本数）
lr = config["train"]["lr"]  # 学习率（优化器的步长）
epochs = config["train"]["epochs"]  # 训练轮数（完整遍历训练集的次数）
model_path = config["model"]["path"]  # 模型保存路径
data_root = config["data"]["root"]  # 数据集存放的根目录

# 检测可用的计算设备：如果有 GPU 则使用 CUDA，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================
# 3. 加载数据集
# ======================

# 加载 MNIST 训练集
train_data = dataset.MNIST(
    root=data_root,  # 数据集存放路径
    train=True,  # 加载训练集（True 表示训练集，False 表示测试集）
    transform=transforms.ToTensor(),  # 将 PIL 图像或 numpy 数组转换为 PyTorch 张量，并归一化到 [0, 1]
    download=True  # 如果本地没有数据集，则自动下载
)

# 加载 MNIST 测试集
test_data = dataset.MNIST(
    root=data_root,
    train=False,  # 加载测试集
    transform=transforms.ToTensor(),
    download=True
)

# 创建数据加载器（DataLoader），用于批量加载数据
train_loader = data_utils.DataLoader(
    train_data,  # 数据集
    batch_size=batch_size,  # 每批次的样本数量
    shuffle=True  # 每个 epoch 开始时打乱数据顺序，防止模型记住顺序
)

test_loader = data_utils.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False  # 测试时不需要打乱数据
)

# ======================
# 4. 初始化模型、损失函数和优化器
# ======================

# 实例化数字分类器模型，并将其移动到指定设备（GPU 或 CPU）
model = DigitClassifier().to(device)

# 定义损失函数：交叉熵损失，适用于多分类问题
loss_func = torch.nn.CrossEntropyLoss()

# 定义优化器：Adam 优化器，用于更新模型参数
optimizer = torch.optim.Adam(
    model.parameters(),  # 需要优化的模型参数
    lr=lr  # 学习率
)

# ======================
# 5. 训练循环
# ======================

# 外层循环：遍历每个 epoch（训练轮次）
for epoch in range(epochs):
    # 将模型设置为训练模式（启用 Dropout、BatchNorm 等训练时特有的行为）
    model.train()

    total_loss = 0  # 累计当前 epoch 的总损失

    # 内层循环：遍历训练数据加载器中的每个批次
    # enumerate 返回 (批次索引, (图像数据, 标签))
    for i, (images, labels) in enumerate(train_loader):
        # 将图像和标签移动到指定的计算设备（GPU 或 CPU）
        images, labels = images.to(device), labels.to(device)

        # 前向传播：将图像输入模型，得到预测输出
        outputs = model(images)
        # 计算损失：比较预测输出和真实标签
        loss = loss_func(outputs, labels)

        # 反向传播前清零梯度（防止梯度累积）
        optimizer.zero_grad()
        # 反向传播：计算梯度
        loss.backward()
        # 更新模型参数（根据梯度和学习率）
        optimizer.step()

        # 累加当前批次的损失值
        total_loss += loss.item()  # .item() 将张量转换为 Python 数值

    # 打印当前 epoch 的平均训练损失
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

    # ======================
    # 6. 测试（验证）循环
    # ======================

    # 将模型设置为评估模式（禁用 Dropout、使用 BatchNorm 的全局统计量等）
    model.eval()

    test_loss = 0  # 累计测试损失
    correct = 0  # 累计正确预测的样本数

    # 禁用梯度计算（减少内存消耗，加速推理）
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for images, labels in test_loader:
            # 调用自定义的评估函数，返回损失值和正确预测数
            loss_val, correct_val = eval_step(
                model,  # 模型
                images,  # 图像数据
                labels,  # 标签
                loss_func,  # 损失函数
                device  # 计算设备
            )
            test_loss += loss_val  # 累加损失
            correct += correct_val  # 累加正确数

    # 计算测试集上的准确率（正确预测数 / 总样本数）
    acc = correct / len(test_data)

    # 打印测试损失和准确率
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")

# ======================
# 7. 保存训练好的模型
# ======================

# 将模型的 state_dict（所有可学习参数）保存到指定路径
torch.save(model.state_dict(), model_path)