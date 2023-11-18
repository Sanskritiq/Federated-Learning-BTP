import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)
from configs.conf_loader import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import trange

from functions.evaluation import EvaluationMetrics


class ClientEvaluator:
    def __init__(self, client_id, global_model: nn.Module, test_loader: DataLoader) -> None:
        self.client_id = client_id
        self.model = global_model
        self.test_loader = test_loader
        self.epochs = len(self.test_loader)
        self.iterator = iter(test_loader)
        self.accuracy = 0
        self.loss = 0
        self.conf_mat = torch.zeros([num_classes, num_classes])

    def test(self):
        with torch.no_grad():
            self.model.eval()
            for _ in trange(self.epochs, desc="client [{}]".format(self.client_id)):
                x, y_true = next(self.iterator)
                y_pred = self.model(x)
                evaluation = EvaluationMetrics(y_pred, y_true)
                self.accuracy += evaluation.accuracy()
                self.loss += evaluation.loss()
                conf_mat = torch.from_numpy(evaluation.conf_matrix())
                self.conf_mat = torch.add(self.conf_mat, conf_mat)
            self.accuracy /= self.epochs
            self.loss /= self.epochs
        self.iterator = iter(self.test_loader)


if __name__ == '__main__':
    from models.models import CNN_MNIST
    from dataset.preprocess import Preprocess
    model = CNN_MNIST()
    pre = Preprocess()
    pre.client_preprocess_train()
    trainset = pre.client_train_sets
    evaluator = ClientEvaluator(0, model, trainset[0])
    evaluator.test()
    print(evaluator.accuracy)
    print(evaluator.loss)
    print(evaluator.conf_mat)