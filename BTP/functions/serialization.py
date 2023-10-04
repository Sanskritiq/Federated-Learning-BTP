from torch import nn
import torch

class Serialization:
    def __init__(self):
        pass
    
    @staticmethod
    def serialize(model: nn.Module) -> torch.Tensor:
        params = [param.data.view(-1) for param in model.parameters()]
        params = torch.cat(params)
        return params
    
    @staticmethod
    def deserialize(model: nn.Module, serialized_params:torch.Tensor) -> None:
        index = 0
        for param in model.parameters():
            numel = param.data.numel()
            size = param.data.size()
            param.data.copy_(serialized_params[index:index+numel].view(size))
            index += numel
            
if __name__=='__main__':
    import sys
    import os
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(sys_path)
    from models.models import ClassicCNN
    model = ClassicCNN()
    tool = Serialization().serialize
    print(tool(model))