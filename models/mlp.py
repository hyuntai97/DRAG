import torch 
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}

# image 
class ImageMLP(nn.Module):
    def __init__(self, mlp_hidden_dim=128, output_dim=1, num_layers=2, flat_dim=64*3*3, activation='relu'):
        super().__init__()

        act = ACTIVATIONS[activation](inplace=True)
        model = []
        input_dim = flat_dim 
        for _ in range(num_layers-1):
            model.append(nn.Linear(input_dim, mlp_hidden_dim, bias=False))
            model.append(act)
            input_dim = mlp_hidden_dim 
            mlp_hidden_dim = mlp_hidden_dim // 2 
            
        model.append(nn.Linear(input_dim, output_dim, bias=False))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x 
    
# tabular 
class TabularMLP(nn.Module):
    def __init__(self, mlp_hidden_dim=32, input_dim=64, output_dim=1, num_layers=2, activation='relu', nl_mlp=0):
        super().__init__()

        act = ACTIVATIONS[activation](inplace=True)
        model = []

        for _ in range(num_layers-1):
            model.append(nn.Linear(input_dim, mlp_hidden_dim, bias=False))
            if nl_mlp == 1:
                model.append(act)
            input_dim = mlp_hidden_dim 
            mlp_hidden_dim = mlp_hidden_dim // 2 
            
        model.append(nn.Linear(input_dim, output_dim, bias=False))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x 
    