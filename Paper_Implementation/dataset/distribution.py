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
        # Generate a dictionary of sample indexes for each client based on the Dirichlet distribution
        for client_id in self.data_distribution:
            num_samples = len(dataset)
            samples_per_class = [int(self.data_distribution[client_id][i] * num_samples) for i in range(self.num_classes)]
            client_sampler = []
            for class_idx, num_samples in enumerate(samples_per_class):
                class_samples = np.random.choice(np.where(dataset.targets == class_idx)[0], num_samples, replace=False)
                client_sampler.extend(class_samples)
            client_samplers[client_id] = client_sampler
        
        return client_samplers

if __name__=="__main__":
    d = Distribution()
    d.generate_dirichlet_distribution()
    d.plot_distribution()