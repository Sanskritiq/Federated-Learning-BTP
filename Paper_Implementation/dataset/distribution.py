import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.conf_loader import *

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class Distribution:
    def __init__(self):
        self.data_distribution = {client_ids[i]: 0 for i in range(num_clients)}
        self.client_samplers = {client_ids[i]: [] for i in range(num_clients)}
    
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
        plt.show()
        
    def get_sampler(self, dataset: Dataset):
        self.generate_dirichlet_distribution()
        # Generate a dictionary of sample indexes for each client based on the Dirichlet distribution
        
        num_samples_per_class = np.unique(dataset.targets, return_counts=True)[1]
        indexes_per_label = [np.where(dataset.targets == i)[0] for i in range(num_classes)]
        
        for client_id in self.data_distribution:
            samples_per_class = [int(self.data_distribution[client_id][i] * num_samples_per_class[i]) for i in range(num_classes)]
            client_sampler = []
            for i, num_samples in enumerate(samples_per_class):
                class_samples = np.random.choice(indexes_per_label[i], samples_per_class[i], replace=False)
                client_sampler.extend(class_samples)
            self.client_samplers[client_id] = client_sampler

if __name__=="__main__":
    from torchvision.datasets import MNIST, CIFAR10
    from torchvision import transforms
    d = Distribution()
    d.generate_dirichlet_distribution()
    # d.plot_distribution()
    train = MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    print(train)
    d.get_sampler(train)
    print(len(d.client_samplers[0]), len(d.client_samplers[1]))
    d.plot_distribution()
    # print(len(d.client_samplers[0][0]), len(d.client_samplers[1][0]))
    