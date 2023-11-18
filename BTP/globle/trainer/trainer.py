import torch
from torch import nn
from tqdm import trange
import random

import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)

from configs.confLoader import *
from fedLearn.fedLearn import fedratedLearning
from client.trainer.trainer import ClientTrainer
from data.dataLoader import ClientPreprocessTrain
from functions.serialization import Serialization

class GlobalTrainer:
    def __init__(self, global_model: nn.Module, fed_learn_method = 'fedavgM') -> None:
        self.global_model = global_model
        self.epochs = global_epochs
        self.client_ids = [i for i in range(n_clients)]
        self.trainset = ClientPreprocessTrain()
        self.client_trainer_set = {}
        self.fed_learn_method = fed_learn_method
        agg_dict = {'fedavg': fedratedLearning.aggregate, 'fedavgM': fedratedLearning.aggregate_with_momentum}
        self.aggregator = agg_dict[self.fed_learn_method]
        self.velocity = torch.zeros(1)
        for client_id in self.client_ids:
            self.client_trainer_set[client_id] = ClientTrainer(client_id, global_model, self.trainset[client_id])
            
    def train(self):        
        for _ in trange(self.epochs, desc="global_training_epoch"):
            rand_client_ids = random.sample(self.client_ids, k=int(n_clients*random_select_ratio))
            weights, params = [], []
            for client_id in rand_client_ids:
                serialized_global_params = Serialization.serialize(self.global_model)
                self.client_trainer_set[client_id].train(serialized_global_params)
                w, p = self.client_trainer_set[client_id].get_weight_param()
                weights.append(w)
                params.append(p)
            global_params = torch.cat([param.data.clone().view(-1) for param in self.global_model.parameters()])
            self.serialized_params, self.velocity = self.aggregator(params, weights, global_params, self.velocity)
            print(self.velocity)
            Serialization.deserialize(self.global_model, self.serialized_params)

        
                       