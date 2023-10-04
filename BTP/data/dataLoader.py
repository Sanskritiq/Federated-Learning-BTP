from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.confLoader import *


def iid(indexes, n):
    random.shuffle(indexes)

    part_size = len(indexes) // n
    remainder = len(indexes) % n

    start = 0
    end = 0
    parts = {}

    for i in range(n):
        end += part_size + (1 if i < remainder else 0)
        part = indexes[start:end]
        parts[i] = part
        start = end

    return parts

def non_iid(indexes, n, targets):
    index_label_pairs = [(index, target) for index, target in zip(indexes, targets)]
    index_label_pairs_least = index_label_pairs[:client_least_data*n_clients]
    index_label_pairs = index_label_pairs[client_least_data*n_clients:]
    index_label_pairs.sort(key=lambda x: x[1])
    least_indexes = [index for index, _ in index_label_pairs_least]
    parts = iid(least_indexes, n)
    
    part_size = len(indexes) // n
    remainder = len(indexes) % n

    start = 0
    end = 0
    
    for i in range(n):
        end += part_size + (1 if i < remainder else 0)
        part = [index for index, _ in index_label_pairs[start:end]]
        parts[i] += part
        start = end

    return parts

    
def unequal(indexes, n):
    random.shuffle(indexes)
    least_indexes = indexes[:client_least_data*n_clients]
    indexes = indexes[client_least_data*n_clients:]
    parts = iid(least_indexes, n)

    part_size = len(indexes) // n
    remainder = len(indexes) % n

    start = 0
    end = 0

    for i in range(n):
        part_size = random.randint(1, len(indexes) // (n - i))
        end += part_size + (1 if i < remainder else 0)
        part = indexes[start:end]
        parts[i] += part
        start = end

    return parts
    

def ClientPreprocessTrain():
    if dataset_name == 'mnist':
        train = MNIST(dataset_root_path, train=True, transform=transforms.ToTensor(), download=True)
    elif dataset_name == 'cifar10':
        train = CIFAR10(dataset_root_path, train=True, transform=transforms.ToTensor(), download=True)
    else:
        train = MNIST(dataset_root_path, train=True, transform=transforms.ToTensor(), download=True)
    
    client_ids = [i for i in range(n_clients)]
    trainsets = {}
    if client_data_distribution_type == 'unequal':
        train_samplers = unequal([i for i in range(len(train.targets))], n_clients)
    elif client_data_distribution_type == 'non-iid':
        train_samplers = non_iid([i for i in range(len(train.targets))], n_clients, train.targets)
    else:
        train_samplers = iid([i for i in range(len(train.targets))], n_clients)
    
    for client_id in client_ids:
        trainsets[client_id] = DataLoader(train, client_train_batch_size, shuffle=False, sampler=train_samplers[client_id])
    
    return trainsets

def ClientPreprocessTest():
    if dataset_name == 'mnist':
        test = MNIST(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
    elif dataset_name == 'cifar10':
        test = CIFAR10(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
    else:
        test = MNIST(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
    
    client_ids = [i for i in range(n_clients)]
    testsets = {}
    if client_data_distribution_type == 'unequal':
        test_samplers = unequal([i for i in range(len(test.targets))], n_clients)
    elif client_data_distribution_type == 'non-iid':
        test_samplers = non_iid([i for i in range(len(test.targets))], n_clients, test.targets)
    else:
        test_samplers = iid([i for i in range(len(test.targets))], n_clients)
    
    for client_id in client_ids:
        testsets[client_id] = DataLoader(test, client_test_batch_size, shuffle=False, sampler=test_samplers[client_id])
    
    return testsets

def GlobalPreprocess():
    if dataset_name == 'mnist':
        test = MNIST(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
    elif dataset_name == 'cifar10':
        test = CIFAR10(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
    else:
        test = MNIST(dataset_root_path, train=False, transform=transforms.ToTensor(), download=True)
        
    testset = DataLoader(test, global_batch_size)
    return testset  
    
    
    
    
if __name__=='__main__':
    client_train, client_test = ClientPreprocessTrain(), GlobalPreprocess()
    global_test = GlobalPreprocess()
    print(len(client_test))
    print(len(client_train))
    data, target = next(iter(client_train[0]))
    print(target)
    print(data.shape)
    print(len(global_test))
    print(global_test.dataset.targets[0])
        