import torch
import matplotlib.pyplot as plt
import config
from data import get_dataloaders
from model import CIFAR10CNN

model = CIFAR10CNN().to(config.DEVICE)
model.load_state_dict(torch.load(config.CHECKPOINT, weights_only=True))

train_loader, val_loader, test_loader = get_dataloaders()

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


images, labels = next(iter(test_loader))  # 取第一批
first_label = labels[0]                   # 对应标签
img = images[0].cpu()
img = img * std + mean
img = img.clamp(0, 1)
img = img.permute(1, 2, 0)

plt.imshow(img)
plt.title(config.CLASSES[first_label.item()])
plt.axis("off")
plt.show()
