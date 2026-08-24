import torch
import matplotlib.pyplot as plt
import config
from data import get_test_loader
from model import CIFAR10CNN


def main():
    model = CIFAR10CNN().to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.CHECKPOINT_BEST, weights_only=True, map_location=config.DEVICE)
    )

    model.eval()

    test_loader = get_test_loader()

    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)

    images, labels = next(iter(test_loader))
    x = images[0:1].to(config.DEVICE)
    with torch.no_grad():
        pred = model(x)
    pred_idx = pred.argmax(1).item()
    actual_idx = labels[0].item()
    img = images[0].cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"pred: {config.CLASSES[pred_idx]}, actual: {config.CLASSES[actual_idx]}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
