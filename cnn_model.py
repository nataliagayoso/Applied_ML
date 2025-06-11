# cnn_model.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def load_cnn_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicCNN()
    model_path = Path(__file__).resolve().parent / "models" / "cnn_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE), DEVICE
