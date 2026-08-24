import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from src import config
from src.data import get_test_loader
from src.utils import denormalize, load_model, predict_index


def main():
    model = load_model(config.CHECKPOINT_BEST)
    test_loader = get_test_loader()

    images, labels = next(iter(test_loader))
    pred_idx = predict_index(model, images[0])
    actual_idx = labels[0].item()

    plt.imshow(denormalize(images[0]))
    plt.title(f"pred: {config.CLASSES[pred_idx]}, actual: {config.CLASSES[actual_idx]}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
