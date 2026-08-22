import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import DetectionLoss
from model import DetectionModel
from yolo_dataset import YOLODATASET
from tqdm import tqdm

model = DetectionModel()
loss_function = DetectionLoss()
yolo_datasets = YOLODATASET(
    r"D:\code\yolo\ultralytics-main\datasets\bvn\images\train",
r"D:\code\yolo\ultralytics-main\datasets\bvn\labels\train",
    transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]),
    None
)

dataloader = DataLoader(yolo_datasets, batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in tqdm(range(3), desc="Training"):
    model.train()
    loss_value = 0.0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        predicts = model(images)
        loss = loss_function(predicts, targets)
        loss_value += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tqdm.write(f"Epoch {epoch + 1} loss: {loss_value:.4f}")
torch.save(model.state_dict(), "yolo.pth")