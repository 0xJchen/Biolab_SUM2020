import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

bottleneck_dim=5

GPU_NUM = 5
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform, download=True)
testset = MNIST('./data', transform=img_transform, download=True, train=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:5'
    else:
        device = 'cpu'
    return device


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, bottleneck_dim))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class new_autoencoder(nn.Module):
    def __init__(self):
        super(new_autoencoder, self).__init__()
        self.upper_encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, bottleneck_dim))

        self.lower_encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, bottleneck_dim))

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim*2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        c = self.upper_encoder(x)
        f = self.lower_encoder(x)
        combined = torch.cat((c.view(c.size(0), -1),
                              f.view(f.size(0), -1)), dim=1)
        out = self.decoder(combined)
        return out


def test_image_reconstruction(net1, net2, testloader):
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs1 = net1(img)
        outputs2 = net2(img)
        outputs1 = outputs1.view(outputs1.size(0), 1, 28, 28).cpu().data
        outputs2 = outputs2.view(outputs2.size(0), 1, 28, 28).cpu().data
        save_image(outputs1, './test/{}_reconstruction.png'.format(bottleneck_dim))
        save_image(outputs2, './test/{}_reconstruction.png'.format(bottleneck_dim*2))
        img = img.view(img.size(0), 1, 28, 28).cpu().data
        save_image(img, './test/{}_raw.png'.format(bottleneck_dim*2))
        break


device = get_device()
# test with old output
model = autoencoder()
model.load_state_dict(torch.load('./weight/sim_autoencoder.pth'))
model.to(device)

# test with new output
new_model = new_autoencoder()
new_model.load_state_dict(torch.load('./weight/new_autoencoder.pth'))
new_model.to(device)

test_image_reconstruction(model, new_model, testloader)