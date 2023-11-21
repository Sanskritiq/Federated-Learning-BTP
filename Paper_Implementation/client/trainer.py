import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from copy import deepcopy
from tqdm import trange

import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)
from configs.conf_loader import *
from functions.serialization import Serialization


class ClientTrainer:
    def __init__(self, client_id, global_model:nn.Module, train_loader:DataLoader) -> None:
        # self.global_model = global_model
        self.client_id = client_id
        self.model = deepcopy(global_model)
        self.train_loader = train_loader
        self.iterator = iter(self.train_loader)
        self.lr = client_lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.epochs = client_epochs
        self.loss_history = []
        
    def train(self, serialized_global_params: torch.Tensor = None):
        if serialized_global_params is not None:
            Serialization.deserialize(self.model, serialized_global_params)
        self.model.train()

        for _ in trange(self.epochs, desc="client [{}]".format(self.client_id)):
            x, y_true = self.trainLoaderIterator()
            y_pred = self.model(x)
            loss_value = self.loss(y_pred, y_true)

            if aggregator_method == 'fedprox':
                # Initialize gradients
                self.optimizer.zero_grad()
                # Add proximal term to loss (FedProx)
                w_diff = torch.tensor(0.)
                for w, w_t in zip(self.model.parameters(), serialized_global_params):
                    w_diff += torch.pow(torch.norm(w.data - w_t), 2)
                    # Create a dummy tensor that depends on w to perform backward pass
                    dummy_loss = torch.sum(w)
                    dummy_loss.backward()
                    # Now w.grad should not be None
                    w.grad.data += proximal_coff * (w_t - w.data)
                loss_value += proximal_coff / 2. * w_diff

            loss = loss_value.item()
            self.loss_history.append(loss)
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # Ensure w.grad is not None after backward pass
            for w in self.model.parameters():
                assert w.grad is not None, "Gradient is None after backward pass!"
        
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
    from models.models import CNN_MNIST
    from dataset.preprocess import Preprocess
    model = CNN_MNIST()
    pre = Preprocess()
    pre.client_preprocess_train()
    trainset = pre.client_train_sets
    trainer = ClientTrainer(0, model, trainset[0])
    trainer.train()
    w, p = trainer.get_weight_param()
    print(w)
    print(p)
    pre.client_preprocess_test()
    testset = pre.client_test_sets
    trainer = ClientTrainer(0, model, testset[0])
    trainer.train()
    w, p = trainer.get_weight_param()
    print(w)
    print(p)