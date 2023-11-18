import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.conf_loader import *

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

class Distribution:
    def __init__(self, dataset: Dataset):
        self.client_samplers = {client_ids[i]: [] for i in range(num_clients)}
        self.data_distribution = {client_ids[i]: 0 for i in range(num_clients)}
        self.dataset = dataset
        
    def plot_distribution(self):
        pass
    
    def get_sampler(self):
        pass
    

class DirichletDistribution(Distribution):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
    
    def generate_dirichlet_distribution(self):
        # generate dirichlet distribution
        dirichlet_distribution = np.random.dirichlet(np.repeat(dirichlet_alpha, num_classes), num_clients)
        for i in range(num_clients):
            self.data_distribution[client_ids[i]] = dirichlet_distribution[i]
            
    def plot_distribution(self):
        # plot the distribution
        plt.figure(figsize=(10, 5))
        for i in range(num_classes):
            plt.bar(client_ids, [self.data_distribution[client_id][i] for client_id in client_ids], bottom=[sum(self.data_distribution[client_id][:i]) for client_id in client_ids], label='class {}'.format(i))
        # plt.legend()
        plt.savefig('dirichlet_distribution.png')
        
    def get_sampler(self):
        self.generate_dirichlet_distribution()
        # Generate a dictionary of sample indexes for each client based on the Dirichlet distribution
        
        num_samples_per_class = np.unique(self.dataset.targets, return_counts=True)[1]
        indexes_per_label = [np.where(self.dataset.targets == i)[0] for i in range(num_classes)]
        
        for client_id in self.data_distribution:
            samples_per_class = [int(self.data_distribution[client_id][i] * num_samples_per_class[i]) for i in range(num_classes)]
            client_sampler = []
            for i, num_samples in enumerate(samples_per_class):
                class_samples = np.random.choice(indexes_per_label[i], samples_per_class[i], replace=False)
                client_sampler.extend(class_samples)
            self.client_samplers[client_id] = client_sampler
            
            
class IIDDistribution(Distribution):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        
    def get_sampler(self):
        indexes = list(range(len(self.dataset.targets)))
        random.shuffle(indexes)

        part_size = len(indexes) // num_clients
        remainder = len(indexes) % num_clients

        start = 0
        end = 0

        for i in range(num_clients):
            end += part_size + (1 if i < remainder else 0)
            part = indexes[start:end]
            self.client_samplers[client_ids[i]] = part
            start = end
            
    def plot_distribution(self):
        for client_id in self.client_samplers:
            self.data_distribution[client_id] = len(self.client_samplers[client_id]) / len(self.dataset.targets)
        # plot the distribution
        plt.figure(figsize=(10, 5))
        plt.bar(client_ids, [self.data_distribution[client_id] for client_id in client_ids])
        plt.savefig('iid_distribution.png')

if __name__=="__main__":
    from torchvision.datasets import MNIST, CIFAR10
    from torchvision import transforms
    train = MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    print(train)
    d = DirichletDistribution(train)
    d.get_sampler()
    print(len(d.client_samplers[0]), len(d.client_samplers[1]))
    d.plot_distribution()
    # print(len(d.client_samplers[0][0]), len(d.client_samplers[1][0]))
    
    d = IIDDistribution(train)
    d.get_sampler()
    print(len(d.client_samplers[0]), len(d.client_samplers[1]))
    d.plot_distribution()
    