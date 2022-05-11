import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1  = nn.Linear(2560, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3  = nn.Linear(20, 400)
        self.fc4  = nn.Linear(400, 2560)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def gaussian_hyperspheric_offset(self, mu, std, n_dim, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        # sample_mean = torch.zeros(n_dim)
        # sample_var = torch.eye(n_dim,n_dim)
        #
        # sampler = MultivariateNormal(sample_mean,sample_var)
        s = torch.randn(n_dim)
        n = torch.randn(n_dim)
        s_norm = torch.norm(s)
        s = s/s_norm
        return mu*s+std*n

    def reparameterize_gho(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        n_dim = mu.shape[1]
        return self.gaussian_hyperspheric_offset(mu,std,n_dim=n_dim)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# sigmoid >0 可以用kld
class Uni_Feature(nn.Module):
    def __init__(self):
        super(Uni_Feature, self).__init__()
        self.fc1  = nn.Linear(20, 200)
        self.fc2 = nn.Linear(200, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Generator_global_distribution(nn.Module):
    def __init__(self):
        super(Generator_global_distribution, self).__init__()
        self.fc1  = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,20)
        self.act = F.relu
        self.act_output = F.sigmoid
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.act_output(x)