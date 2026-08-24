from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import config
from .model import CIFAR10CNN


def ensure_output_dirs() -> None:
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint: Path = config.CHECKPOINT_BEST) -> CIFAR10CNN:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `python train.py` first to train and save weights."
        )
    model = CIFAR10CNN().to(config.DEVICE)
    model.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location=config.DEVICE)
    )
    model.eval()
    return model


def evaluate(model: CIFAR10CNN, loader: DataLoader) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            pred = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Convert normalized CHW tensor to HWC float image in [0, 1]."""
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    img = image.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0)


def predict_index(model: CIFAR10CNN, image: torch.Tensor) -> int:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    with torch.no_grad():
        pred = model(image.to(config.DEVICE))
    return pred.argmax(1).item()
