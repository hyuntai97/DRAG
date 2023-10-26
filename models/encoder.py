import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#-- LeNet based image encoder
class LeNet(nn.Module):

    def __init__(self, num_channels=1, latent_dim=16):
        super(LeNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(num_channels, latent_dim, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(latent_dim, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(latent_dim, latent_dim*2, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(latent_dim*2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(latent_dim*2, latent_dim*4, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(latent_dim*4, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        return x, x
    

#-- Tabnet based tabular encoder 
class MLP_Feature_Extractor(nn.Module):
    def __init__(self,
                 input_dim=2, 
                 num_hidden_nodes=20):
        super(MLP_Feature_Extractor, self).__init__()
        self.input_dim = input_dim
        self.num_hidden_nodes = num_hidden_nodes
        activ = nn.ReLU(True)
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ),
            ('fc2', nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)),
            ('relu2', activ),
            ('fc3', nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)),
            ('relu3', activ),
            ('fc4', nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)),
            ('relu4', activ)
            ]))

    def forward(self, input):
        out = self.feature_extractor(input)
        feat = out.clone().mean([-1])
        return out, feat