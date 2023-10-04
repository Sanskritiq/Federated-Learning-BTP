from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
import torch

import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)

from configs.confLoader import *

class EvaluationMetrics:
    def __init__(self, y_pred, y_true) -> None:
        self.y_pred = torch.argmax(y_pred, dim=1)
        self.y_true = y_true
        self.y_pred_prob = y_pred
        self.labels = [i for i in range(n_classes)]
    
    def conf_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
    
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def loss(self):
        return log_loss(self.y_true, self.y_pred_prob, labels=self.labels)
    
    
if __name__ == '__main__':
    import random
    y_true = torch.Tensor([0, 8, 4, 1, 5, 7, 1, 9, 5, 3, 1, 9, 3, 5, 6, 2])
    y_pred = torch.rand([16, 10])
    
    evaluation = EvaluationMetrics(y_pred, y_true)
    print(evaluation.accuracy())
    print(evaluation.conf_matrix())
    print(evaluation.loss())
