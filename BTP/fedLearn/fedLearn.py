import torch

class fedratedLearning:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def aggregate(serialized_params_list, weights = None) -> torch.Tensor:
        
        if not weights:
            weights = torch.ones((len(serialized_params_list)))
        else:
            weights = torch.tensor(weights)
            
        weights = weights / torch.sum(weights)
        serialized_params = torch.sum(torch.stack(serialized_params_list, dim=-1)*weights, dim=-1)        
        return serialized_params