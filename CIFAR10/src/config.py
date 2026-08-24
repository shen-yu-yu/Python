from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RUNS_DIR = OUTPUT_DIR / "runs" / "cifar10"

CHECKPOINT_RESULT = CHECKPOINT_DIR / "model.pth"
CHECKPOINT_BEST = CHECKPOINT_DIR / "best.pth"

BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
VAL_RATIO = 0.1

# CIFAR-10 normalization (train set statistics)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = len(CLASSES)
