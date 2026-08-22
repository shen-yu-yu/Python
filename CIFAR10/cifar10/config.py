from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR/"data"
CHECKPOINT = ROOT_DIR/"checkpoints"

BATCH_SIZE = 128
NUM_EPOCHES = 20
LEARNING_RETE = 1e3
NUM_WORKERS = 0
VAL_RATIO = 0.1

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

