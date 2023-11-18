import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.conf_loader import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from dataset.distribution import *

class Preprocess:
    def __init__(self, name = 'mnist', dist = 'iid'):
        self.name = name
        self.dist = dist
        self.DATASET = {'mnist': MNIST, 'cifar10': CIFAR10}
        dist_dict= {'iid': IIDDistribution, 'dirichlet': DirichletDistribution}
        self.train_dataset = self.DATASET[name](root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        self.test_dataset = self.DATASET[name](root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
        self.distribution = dist_dict[dist]
        self.client_test_sets = {}
        self.client_train_sets = {}
        self.global_test = None
    
    def client_preprocess_test(self):
        self.distribution_test = self.distribution(self.test_dataset)
        self.distribution_test.get_sampler()
        
        for client_id in client_ids:
            self.client_test_sets[client_id] = DataLoader(self.test_dataset, batch_size, shuffle=False, sampler=self.distribution_test.client_samplers[client_id])
    
    def client_preprocess_train(self):
        self.distribution_train = self.distribution(self.train_dataset)
        self.distribution_train.get_sampler()
        
        for client_id in client_ids:
            self.client_train_sets[client_id] = DataLoader(self.train_dataset, batch_size, shuffle=False, sampler=self.distribution_train.client_samplers[client_id])
    
    def global_preprocess_test(self):
        self.global_test = DataLoader(self.test_dataset, batch_size)
    
if __name__=="__main__":
    p = Preprocess()
    p.client_preprocess_test()
    p.client_preprocess_train()
