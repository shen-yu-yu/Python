import torch
import config
import requests
import os
from torch import nn
from model import NeuralNetWork
from dataset import training_loader

model = NeuralNetWork().to(config.device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), config.lr)

model.train()
size = len(training_loader)

for i in range(config.epoches):
    print(f"epoch: {i}")
    for batch, (X, y) in enumerate(training_loader):
        X, y = X.to(config.device), y.to(config.device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss}, current: {current} / {size * config.batch_size}")
    i += 1


torch.save(model.state_dict(), "model.pth")
print("model save successful")

send_key = os.environ.get("SERVERCHAN_SENDKEY")

requests.get(
    f"https://sctapi.ftqq.com/{send_key}.send",
    params={"title": "训练完成", "desp": "FashionMNIST 训练结束"}
)
