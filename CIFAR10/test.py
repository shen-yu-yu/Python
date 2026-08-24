import torch

import config
from data import get_test_loader
from model import CIFAR10CNN


def main():
    model = CIFAR10CNN().to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.CHECKPOINT_BEST, weights_only=True, map_location=config.DEVICE)
    )

    test_loader = get_test_loader()

    correct = 0
    model.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
        accuracy = correct / len(test_loader.dataset)
        print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
