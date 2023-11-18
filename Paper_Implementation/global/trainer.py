import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)

from configs.conf_loader import *

import torch
from torch import nn
from tqdm import trange
import random

from fedLearn.fedLearn import fedratedLearning
from client.trainer import ClientTrainer
from dataset.preprocess import Preprocess
from functions.serialization import Serialization

class GlobalTrainer:
    def __init__(self, global_model: nn.Module, fed_learn_method = 'fedavgM') -> None:
        self.global_model = global_model
        self.epochs = global_rounds
        # self.client_ids = [i for i in range(n_clients)]
        pre = Preprocess()
        pre.client_preprocess_train()
        self.trainset = pre.client_train_sets
        self.client_trainer_set = {}
        self.fed_learn_method = fed_learn_method
        agg_dict = {'fedavg': fedratedLearning.aggregate, 'fedavgM': fedratedLearning.aggregate_with_momentum}
        self.aggregator = agg_dict[self.fed_learn_method]
        # self.velocity = torch.zeros(1)
        for client_id in client_ids:
            self.client_trainer_set[client_id] = ClientTrainer(client_id, global_model, self.trainset[client_id])
            
    def train(self, momentum = 0.9):        
        for _ in trange(self.epochs, desc="global_training_epoch"):
            rand_client_ids = random.sample(client_ids, k=int(num_clients*random_select_ratio))
            weights, params = [], []
            for client_id in rand_client_ids:
                serialized_global_params = Serialization.serialize(self.global_model)
                self.client_trainer_set[client_id].train(serialized_global_params)
                w, p = self.client_trainer_set[client_id].get_weight_param()
                weights.append(w)
                params.append(p)
            global_params = torch.cat([param.data.clone().view(-1) for param in self.global_model.parameters()])
            self.serialized_params = self.aggregator(params, weights, global_params, momentum)
            # print(self.velocity)
            Serialization.deserialize(self.global_model, self.serialized_params)
            
            
if __name__ == '__main__':
    from models.models import CNN_MNIST
    model = CNN_MNIST()
    trainer = GlobalTrainer(model)
    trainer.train()

        