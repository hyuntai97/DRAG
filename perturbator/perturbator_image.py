import torch
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct

class MultiPerturbator(nn.Module):
    def __init__(self, config, num_channels, num_perturb, flat_dim):
        self.config = config 
        self.num_channels = num_channels
        self.num_perturb = num_perturb 
        self.flat_dim = flat_dim

    def _make_nets(self):
        activation=self.config['model_params']['activation'] or 'relu'
        num_res_blocks=self.config['model_params']['num_res_blocks'] or 5
        p_maxpool=self.config['model_params']['p_maxpool']  
        p_num_layers=self.config['model_params']['p_num_layers']
        p_hidden_dim=self.config['model_params']['p_hidden_dim']

        views = nn.ModuleList(
            [Perturbator(self.num_channels, activation, num_res_blocks, self.flat_dim, p_maxpool, p_num_layers, p_hidden_dim) for _ in range(self.num_perturb)]
        )

        return views

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}

class Perturbator(torch.nn.Module):
    def __init__(self, num_channels=3, activation='relu',  num_res_blocks=3, flat_dim=64*3*3, p_maxpool=True, p_num_layers=3, p_hidden_dim=16):
        super().__init__()
        
        self.flatten_size = flat_dim

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        act = ACTIVATIONS[activation]()

        model = []

        # Initial convolution layers (+ 1 for noise filter)
        input_dim = num_channels + 1
        for i in range(p_num_layers-1):
            if i == 0:
                model.append(ConvLayer(input_dim, p_hidden_dim, kernel_size=9, stride=1))
            else: 
                model.append(ConvLayer(input_dim, p_hidden_dim, kernel_size=3, stride=2))
            model.append(torch.nn.InstanceNorm2d(p_hidden_dim, affine=True))
            model.append(act)
            
            input_dim = p_hidden_dim
            p_hidden_dim = p_hidden_dim * 2 

        model.append(ConvLayer(input_dim, p_hidden_dim, kernel_size=3, stride=2))
        if p_maxpool == "True":
            model.append(nn.MaxPool2d(2,2))

        self.model = nn.Sequential(*model)

        # non-linear head 
        self.non_linear_head = nn.Sequential(
            nn.Linear(self.flatten_size, self.flatten_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.flatten_size, self.flatten_size)
        )

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
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.model(y)
        
        # Features that could be useful for other auxilary layers / losses.
        features = self.non_linear_head(y.flatten(start_dim=1))

        return y, features


    def forward(self, x):
        y = x

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)

        return y_pixels, features

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
