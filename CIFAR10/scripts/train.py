import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.data import get_train_val_loader
from src.model import CIFAR10CNN
from src.utils import ensure_output_dirs


def main():
    ensure_output_dirs()
    train_loader, val_loader = get_train_val_loader()
    model = CIFAR10CNN().to(config.DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter(log_dir=str(config.RUNS_DIR))

    best_acc = 0.0

    for i in range(config.NUM_EPOCHS):
        loss_value = 0.0
        correct = 0
        accuracy = 0.0
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
        print(f"epoch: {i + 1}, avg_loss: {loss_value / len(train_loader)}")
        writer.add_scalar("Loss/train", (loss_value / len(train_loader)), global_step=(i + 1))
        model.eval()
        with torch.no_grad():
            for batch, (images, labels) in enumerate(val_loader):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                pred = model(images)
                correct += (pred.argmax(1) == labels).sum().item()
            accuracy = correct / len(val_loader.dataset)
            print(f"epoch: {i + 1}, accuracy: {accuracy}")
            writer.add_scalar("Accuracy/val", accuracy, global_step=(i + 1))
        print("=========================================")
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), config.CHECKPOINT_BEST)

    writer.close()
    torch.save(model.state_dict(), config.CHECKPOINT_RESULT)


if __name__ == "__main__":
    main()
