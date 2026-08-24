import torch

import config
from data import get_dataloaders
from model import CIFAR10CNN

model = CIFAR10CNN().to(config.DEVICE)
model.load_state_dict(torch.load(config.CHECKPOINT, weights_only=True))

train_loader, val_loader, test_loader = get_dataloaders()

correct = 0
for _, (images, labels) in enumerate(test_loader):
    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
    pred = model(images)
    correct += (pred.argmax(1) == labels).sum().item()
accuracy = correct / (len(test_loader) * config.BATCH_SIZE)
print(f"accuracy: {accuracy}")