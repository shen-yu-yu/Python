import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import config

def get_transforms():
    train_transformers = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]
    )

    eval_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]
    )

    return train_transformers, eval_transformer

def get_test_loader():
    train_transformers, eval_transformers = get_transforms()
    test_set = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=eval_transformers
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return test_loader

def get_train_val_loader():

    train_transformers, eval_transformers = get_transforms()

    train_source = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transformers
    )

    val_source = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=eval_transformers
    )

    total = len(train_source)
    val_size = int(total * config.VAL_RATIO)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = Subset(train_source, train_indices)
    val_set = Subset(val_source, val_indices)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader

def main():
    train_loader, val_loader = get_train_val_loader()
    images, labels = next(iter(train_loader))
    print("train batch: ", len(train_loader))
    print("val batch: ", len(val_loader))
    print("image batch shape:", images.shape)
    print("label batch shape:", labels.shape)
    print("label sample:", labels[:8].tolist())
    print("classes:", config.CLASSES)

if __name__ == "__main__":
    main()



