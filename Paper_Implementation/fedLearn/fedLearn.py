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
    def aggregate(global_params, weighted_deltaw_list, momentum = None, velocity = None) -> torch.Tensor:
        
        weighted_deltaw = torch.sum(torch.stack(weighted_deltaw_list, dim=-1), dim=-1)    
        global_params += weighted_deltaw    
        return global_params, None

    @staticmethod
    def aggregate_with_momentum(global_params, weighted_deltaw_list, momentum = 0.9, velocity = 0) -> torch.Tensor:
        weighted_deltaw = torch.sum(torch.stack(weighted_deltaw_list, dim=-1), dim=-1)  
        velocity = momentum*velocity + weighted_deltaw
        global_params += velocity
        
        return global_params, velocity
            
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
