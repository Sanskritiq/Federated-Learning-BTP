import sys
import os
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)

from configs.conf_loader import *
import torch

class fedratedLearning:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def aggregate(serialized_params_list, weights = None, global_params = None, velocity = None) -> torch.Tensor:
        if weights is None:
            weights = torch.ones(len(serialized_params_list))
        else:
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        serialized_params = torch.sum(torch.stack(serialized_params_list, dim=-1)*weights, dim=-1)        
        return serialized_params

    @staticmethod
    def aggregate_with_momentum(serialized_params_list, weights = None, global_params = None, momentum = 0.9) -> torch.Tensor:
        for global_param, client_params in zip(global_params, zip(*serialized_params_list)):
            global_param.data = (1 - momentum) * global_param.data + momentum * sum(client_param.data / len(serialized_params_list) for client_param in client_params)

        return global_params
            
if __name__ == '__main__':
    from models.models import CNN_MNIST
    from functions.serialization import Serialization
    local_model1 = CNN_MNIST()
    local_model2 = CNN_MNIST()
    local_model3 = CNN_MNIST()
    tool = Serialization().serialize
    params_list = []
    params1 = tool(local_model1)
    params2 = tool(local_model2)
    params3 = tool(local_model3)
    params_list = [params1, params2, params3]
    weights = [1, 2, 3]
    global_model = CNN_MNIST()
    global_params = tool(global_model)
    updated_global_params1 = fedratedLearning().aggregate(params_list)
    updated_global_params2 = fedratedLearning().aggregate_with_momentum(params_list, None, global_params)
    print(updated_global_params1.shape)
    print(updated_global_params2.shape)
    print(updated_global_params2)
