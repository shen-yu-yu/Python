import torch
import torchvision
import torchvision.models
from torch import nn


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extract = torchvision.models.vgg16().features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        x = self.features_extract(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':

    image = torch.randn(1, 3, 448, 448)

    model = DetectionModel()

    outputs = model(image)
    print(outputs.shape)
