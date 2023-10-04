import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from copy import deepcopy
from tqdm import trange

import sys
import os
sys_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)

from configs.confLoader import *
from functions.serialization import Serialization

class ClientTrainer:
    def __init__(self, client_id, global_model:nn.Module, train_loader:DataLoader) -> None:
        self.client_id = client_id
        self.model = deepcopy(global_model)
        self.train_loader = train_loader
        self.iterator = iter(self.train_loader)
        self.lr = client_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.epochs = client_epochs
        self.loss_history = []
        
    def train(self, serialized_global_params:torch.Tensor = None):
        if serialized_global_params is not None:
            Serialization.deserialize(self.model, serialized_global_params)
        self.model.train()
        
        for _ in trange(self.epochs, desc="client [{}]".format(self.client_id)):
            x, y_true = self.trainLoaderIterator()
            y_pred = self.model(x) 
            loss_value = self.loss(y_pred, y_true)
            loss = loss_value.item()
            self.loss_history.append(loss)
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
        
    def trainLoaderIterator(self):
        try:
            next_batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.train_loader)
            next_batch = next(self.iterator)
        return next_batch
    
    def get_weight_param(self):
        params = Serialization.serialize(self.model)
        weights = len(self.train_loader.dataset)
        
        return weights, params
    
if __name__=='__main__':
    from models.models import ClassicCNN
    from data.dataLoader import ClientPreprocessTrain
    model = ClassicCNN()
    trainset = ClientPreprocessTrain()
    trainer = ClientTrainer(0, model, trainset[0])
    trainer.train()
    w, p = trainer.get_weight_param()
    print(w)
    print(p)