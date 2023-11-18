import torch

class fedratedLearning:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def aggregate(serialized_params_list, weights = None, global_params_list = None, velocity = None) -> torch.Tensor:
        
        if not weights:
            weights = torch.ones((len(serialized_params_list)))
        else:
            weights = torch.tensor(weights)
            
        weights = weights / torch.sum(weights)
        serialized_params = torch.sum(torch.stack(serialized_params_list, dim=-1)*weights, dim=-1)        
        return serialized_params, 0

    @staticmethod
    def aggregate_with_momentum(serialized_params_list, weights = None, global_params_list = None, velocity = None, momentum = 0.9) -> torch.Tensor:
        aggregated_params = global_params_list
        if not weights:
            weights = torch.ones((len(serialized_params_list)))
        else:
            weights = torch.tensor(weights)
            
        weights = weights / torch.sum(weights)
        print(len(serialized_params_list))
        print(serialized_params_list)
        delta_w = torch.sum(torch.stack(serialized_params_list, dim=-1)*weights, dim=-1)
        velocity = momentum * velocity + delta_w
        aggregated_params = aggregated_params - velocity
        return aggregated_params, velocity