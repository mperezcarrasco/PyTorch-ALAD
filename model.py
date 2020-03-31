import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sn


class Discriminatorxz(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Discriminatorxz, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.conv1x = layer.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2x = layer.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2x = layer.BatchNorm2d(256)
        self.conv3x = layer.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3x = layer.BatchNorm2d(512)

        # Inference over z
        self.nn1z = layer.Conv2d(z_dim, 512, 1, stride=1, padding=1)
        self.nn2z = layer.Conv2d(512, 512, 1, stride=1, padding=1)

        # Joint inference
        self.nn1xz = layer.Conv2d(4*4*512 + 512, 1024, 1, stride=1, padding=1)
        self.nn2xz = layer.Conv2d(1024, 1, 1, stride=1, padding=1)

    def inf_x(self, x):
        x = F.leaky_relu(self.conv1x(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2x(self.conv2x(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3x(self.conv3x(x)), negative_slope=0.2)
        return x

    def inf_z(self, z):
        z = F.dropout(F.leaky_relu(self.nn1z(z), negative_slope=0.2), 0.2)
        z = F.dropout(F.leaky_relu(self.nn2z(z), negative_slope=0.2), 0.2)
        return z

    def inf_xz(self, xz):
        intermediate = F.dropout(F.leaky_relu(self.nn1xz(xz), negative_slope=0.2), 0.2)
        xz = self.nn2xz(intermediate)
        return intermediate, xz

    def forward(self, x, z):
        x = self.inf_x(x)
        z = z.view(z.size(0),-1)
        z = self.inf_z(z)
        xz = torch.cat((x.view(x.size(0),-1),z), dim=1)
        intermediate, out = self.inf_xz(xz)
        return torch.sigmoid(out), intermediate


class Discriminatorxx(nn.Module):
    def __init__(self, spectral_norm=False):
        super(Discriminatorxx, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.conv1xx = layer.Conv2d(6, 64, 5, stride=2, padding=1)
        self.conv2xx = layer.Conv2d(64, 128, 5, stride=2, padding=1)
        self.nn3xx = layer.Linear(1, 1)

    def forward(self, x, x_hat):
        xx = torch.cat((x,x_hat), dim=1)
        xx = F.dropout(F.leaky_relu(self.conv1xx(xx), negative_slope=0.2), 0.2)
        xx = F.dropout(F.leaky_relu(self.conv2xx(xx), negative_slope=0.2), 0.2)
        intermediate = xx.view(z.size(0),-1)
        print(intermediate.shape)
        out = self.nn3xx(intermediate)
        return torch.sigmoid(out), intermediate


class Discriminatorzz(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Discriminatorzz, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.nn1zz = layer.Linear(2*z_dim, 32)
        self.nn2zz = layer.Linear(32, 64)
        self.nn3zz = layer.Linear(64, 1)

    def forward(self, z, z_hat):
        zz = torch.cat((z,z_hat), dim=1)
        zz = F.dropout(F.leaky_relu(self.nn1zz(zz), negative_slope=0.2), 0.2)
        intermediate = F.dropout(F.leaky_relu(self.nn2zz(zz), negative_slope=0.2), 0.2)
        out = self.nn3zz(intermediate)
        return torch.sigmoid(out), intermediate


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 3, 4, stride=2)

    def forward(self, z):
        z = z.view(z.size(0),-1)
        z = F.relu(self.bn1(self.deconv1(z)))
        z = F.relu(self.bn2(self.deconv2(z)))
        z = F.relu(self.bn3(self.deconv3(z)))
        z = self.deconv4(z)
        print(z.shape)
        return torch.tanh(z)


class Encoder(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Encoder, self).__init__()
        layer = sn if spectral_norm else nn
        self.z_dim = z_dim
        self.conv1 = layer.Conv2d(3, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = layer.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = layer.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.nn4 = layer.Linear(512*5*5, z_dim*2)

    def reparameterize(self, z):
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.view(x.size(0),-1)
        print(x.shape)
        z = self.reparameterize(self.nn4(x))
        return z.view(x.size(0), self.z_dim, 1, 1)
