import torch
import torch.nn as nn
from torchvision.models import resnet50


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.res = nn.Sequential(
            *(list(resnet50(pretrained=True).children())[:-1])
        )
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.res(x)
        x = x.squeeze()
        x = self.fc(x)
        out = self.sigmoid(x)
        return out
