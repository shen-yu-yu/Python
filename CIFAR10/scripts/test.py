import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.data import get_test_loader
from src.utils import evaluate, load_model


def main():
    model = load_model(config.CHECKPOINT_BEST)
    test_loader = get_test_loader()
    accuracy = evaluate(model, test_loader)
    print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
