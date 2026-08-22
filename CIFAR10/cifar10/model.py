import torch
from torch import nn
from cifar10 import config

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            # block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            # block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 训练时随机丢掉一部分神经元，减轻过拟合
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CIFAR10CNN()
    dummy = torch.randn(8, 3, 32, 32)  # 假装一个 batch
    logits = model(dummy)
    print(model)
    print("input :", dummy.shape)
    print("output:", logits.shape)  # 期望: torch.Size([8, 10])
