# Model that makes prediction of test dataset: 
# This is not part of the fed averaging but an implementation for learning purposes
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import lightning as L
import requests
from PIL import Image
from io import BytesIO


transform=transforms.Compose([
    transforms.RandomRotation(10),      # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

dir_path = os.path.dirname(os.path.abspath(__file__))

# Specify the name of your CSV file
data_dir_path = "Chess/"

# Construct the full file path
data_path = os.path.join(dir_path, data_dir_path)
# get the dataset to be used
dataset=datasets.ImageFolder(root=data_path ,transform=None)

class_names = dataset.classes
print(dataset.classes)

# Class to process the data and divide it into test and train set

class DataProcessor(L.LightningDataModule):
    def __init__(self, transform = transform, batch_size = 32):
        super().__init__()
        self.data_dir = data_path 
        self.transform = transform
        self.batch_size = batch_size

    def getData(self, stage = None):
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        n_data = len(dataset)
        n_train = int(0.8 * n_data) # 80% for training and 20% for testing
        n_test = n_data - n_train

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_dataset

    def test_dataloader(self):
        return self.test_dataset

class TrainingModel(L.LightningModule):

    def __init__(self):
        super(TrainingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)

if __name__ == '__main__':
    datamodule = DataProcessor()
    datamodule.getData()
    model = TrainingModel()
    trainer = L.Trainer(max_epochs=50)
    trainer.fit(model, datamodule)
    datamodule.getData(stage='test')
    test_loader = datamodule.test_dataloader()
    trainer.test(dataloaders=test_loader)
    device = torch.device("cpu")   #"cuda:0"
    model.eval()
    y_true=[]
    y_pred=[]
    true_count = 0
    with torch.no_grad():
        for test_data in datamodule.test_dataloader():
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print(classification_report(y_true,y_pred,target_names=class_names,digits=4))