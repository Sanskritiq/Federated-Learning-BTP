import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.conf_loader import *

import torch
import torch.nn as nn

class MNIST_MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=200, num_classes=10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # Calculate the input size based on your architecture
        self.relu3 = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model_dict = {
    'mnist-mlp': MNIST_MLP,
    'cnn-mnist': CNN_MNIST
}
model = model_dict[model_name]()
    
    
if __name__=="__main__":
    # Define the model
    # input_size = 28 * 28  # MNIST images are 28x28 pixels
    # hidden_size = 200
    # num_classes = 10  # MNIST has 10 classes (digits 0-9)

    # model = MNIST_MLP(input_size, hidden_size, num_classes)

    # # You can print the model to see the architecture
    # print(model)
    from torchvision.datasets import MNIST
    from torch.utils.data import transforms
    train = MNIST(dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    train = DataLoader(train, batch_size=16, shuffle=True)
    # test = torch.rand((1, 1, 28, 28))
    
    model = CNN_MNIST()
    out = model(train)
    print(model)    