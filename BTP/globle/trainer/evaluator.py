import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

import sys
import os
sys_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)
from functions.evaluation import EvaluationMetrics
from configs.confLoader import *


class GlobalEvaluator:
    def __init__(self, global_model: nn.Module, test_loader: DataLoader) -> None:
        self.model = global_model
        self.test_loader = test_loader
        self.epochs = len(self.test_loader)
        self.iterator = iter(test_loader)
        self.accuracy = 0
        self.loss = 0
        self.conf_mat = torch.zeros([n_classes, n_classes])

    def test(self):
        with torch.no_grad():
            self.model.eval()
            for _ in trange(self.epochs, desc="global_evaluation"):
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
    from models.models import ClassicCNN
    from data.dataLoader import GlobalPreprocess
    model = ClassicCNN()
    testset = GlobalPreprocess()
    evaluator = GlobalEvaluator(model, testset)
    evaluator.test()
    print(evaluator.accuracy)
    print(evaluator.loss)
    print(evaluator.conf_mat)
