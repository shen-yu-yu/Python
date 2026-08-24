import torch
import config
from dataset import test_loader
from model import NeuralNetWork


model = NeuralNetWork().to(config.device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

correct = 0
model.eval()
size = len(test_loader)
for _, (X, y) in enumerate(test_loader):
    X, y = X.to(config.device), y.to(config.device)
    pred = model(X)
    correct += (pred.argmax(1) == y).sum().item()

accuracy = correct / size

print(f"accuracy: {accuracy}")