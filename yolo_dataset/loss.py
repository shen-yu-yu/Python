from torch import nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.location_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        predicts_location = predicts[:, 0:4]
        predicts_class = predicts[:, 4:]
        targets_location = targets[:, 0:4].float()
        targets_class = targets[:, 4:].float()

        location_loss_value = self.location_loss(predicts_location, targets_location)
        class_loss_value = self.class_loss(predicts_class, targets_class)

        return location_loss_value + class_loss_value
