

import torch
from torchvision import models


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return x
