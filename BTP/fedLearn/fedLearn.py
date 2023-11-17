import torch

class fedratedLearning:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def aggregate(serialized_params_list, weights = None, global_params_list = None) -> torch.Tensor:
        
        if not weights:
            weights = torch.ones((len(serialized_params_list)))
        else:
            weights = torch.tensor(weights)
            
        weights = weights / torch.sum(weights)
        serialized_params = torch.sum(torch.stack(serialized_params_list, dim=-1)*weights, dim=-1)        
        return serialized_params

    @staticmethod
    def aggregate_with_momentum(serialized_params_list, weights = None, global_params_list = None, momentum = 0.9) -> torch.Tensor:
        aggregated_params = global_params_list
        if not weights:
            weights = torch.ones((len(serialized_params_list)))
        else:
            weights = torch.tensor(weights)
            
        weights = weights / torch.sum(weights)

        # Iterate through the client parameters and weights
        for serialized_params, weight in zip(serialized_params_list, weights):
            # Update the aggregated_params using federated averaging
            aggregated_params = aggregated_params + momentum * (serialized_params - aggregated_params) * weight
        return aggregated_params