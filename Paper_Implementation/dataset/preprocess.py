import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.conf_loader import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

class Preprocess:
    def __init__(self, name = 'mnist'):
        self.name = name
        self.DATASET = {'mnist': MNIST, 'cifar10': CIFAR10}
        self.train_dataset = self.DATASET[name](root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        self.test_dataset = self.DATASET[name](root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    
    def client_preprocess_test(self):
        pass
    
    def client_preprocess_train(self):
        pass
    
    def global_preprocess_test(self):
        pass
    
if __name__=="__main__":
    p = Preprocess()
