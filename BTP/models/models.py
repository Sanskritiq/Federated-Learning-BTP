from torch import nn
import torch

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.confLoader import *

class ClassicCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        return self.net(x)
    
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        return self.net(x)
    
    
if model_name == 'classic_cnn':
    model = ClassicCNN()
elif model_name == 'cnn':
    model = CNN()
else:
    model = ClassicCNN()
    
    
if __name__=='__main__':
    model = CNN()
    x = torch.rand([4, 3, 32, 32])
    print(model)
    out = model.forward(x)
    print(out.shape)