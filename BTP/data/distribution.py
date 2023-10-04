import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.confLoader import *

def data_label_distribution(dataset: DataLoader):
    labels_distribution = {i: 0 for i in range(n_classes)}
    for _, labels in dataset:
        for label in labels:
            labels_distribution[int(label)] += 1
    return labels_distribution
            
if __name__=='__main__':
    from data.dataLoader import ClientPreprocessTrain
    testset = ClientPreprocessTrain()
    dist_1 = data_label_distribution(testset[0])
    print(dist_1)
    