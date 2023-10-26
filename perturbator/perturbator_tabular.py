import torch
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct

class MultiPerturbator(nn.Module):
    def __init__(self, config, num_channels, num_perturb):
        self.config = config 
        self.num_channels = num_channels
        self.num_perturb = num_perturb 

    def _make_nets(self):
        activation=self.config['model_params']['activation'] or 'relu'
        p_hidden_dim=self.config['model_params']['p_hidden_dim'] 

        views = nn.ModuleList(
            [Perturbator(self.num_channels, activation, p_hidden_dim) for _ in range(self.num_perturb)]
        )

        return views

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}

class Perturbator(torch.nn.Module):
    def __init__(self, num_channels=3, activation='relu', latent_dim=24):

        super().__init__()
        
        self.num_channels = num_channels
        self.activation = activation
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = nn.Conv1d(self.num_channels + 1, latent_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=1, bias=True)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        shp = (batch_size, num)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, bound_multiplier=1):
        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.act(self.conv1(y.unsqueeze(-1)))
        y = self.conv2(y)
        
        features = y.clone().mean([-1, -2])

        return y, features


    def forward(self, x):
        y = x

        y_pixels, features = self.basic_net(y, bound_multiplier=1)

        return y_pixels.squeeze(-1), features


