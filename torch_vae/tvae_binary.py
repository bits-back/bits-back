import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions import Normal, Bernoulli
from torchvision.utils import save_image
import numpy as np

torch.manual_seed(17)


class BinaryVAE(nn.Module):
    def __init__(self, hidden_dim=100, latent_dim=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        """Take a z and output a probability (ie Bernoulli param) on each dim"""
        h3 = F.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def loss(self, x):
        z_mu, z_std = self.encode(x.view(-1, 784))
        z = self.reparameterize(z_mu, z_std)  # sample zs
        x_probs = self.decode(z)
        dist = Bernoulli(x_probs)
        l = torch.sum(dist.log_prob(x.view(-1, 784)), dim=1)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / 784.

    def sample(self, epoch, num=64):
        z = torch.randn(num, self.latent_dim)
        x_probs = self.decode(z)
        dist = Bernoulli(x_probs)
        x_sample = dist.sample()
        save_image(x_sample.view(num, 1, 28, 28),
                   'results/epoch_{}_samples.png'.format(epoch))

    def reconstruct(self, x, epoch):
        x = x.view(-1, 784).float()
        z_mu, z_std = self.encode(x)
        z = self.reparameterize(z_mu, z_std)  # sample zs
        x_probs = self.decode(z)
        dist = Bernoulli(x_probs)
        x_recon = dist.sample()
        x_with_recon = torch.cat((x, x_recon))
        save_image(x_with_recon.view(64, 1, 28, 28),
                   'results/epoch_{}_recon.png'.format(epoch))


def train(model, epoch, data_loader, optimizer, log_interval=10):
    model.train()
    losses = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, np.mean(losses)))


def test(model, epoch, data_loader):
    model.eval()
    losses = []
    for data, _ in data_loader:
        loss = model.loss(data)
        losses.append(loss.item())
    print('\nEpoch: {}\tTest loss: {:.6f}\n\n'.format(
        epoch, np.mean(losses)
    ))


if __name__ == '__main__':
    epochs = 20
    batch_size = 100
    randomise_data = True

    model = BinaryVAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    class Randomise:
        def __call__(self, pic):
            return Bernoulli(pic).sample()

    class Round:
        def __call__(self, pic):
            return torch.round(pic)


    if randomise_data:
        binariser = Randomise()
    else:
        binariser = Round()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), binariser])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), binariser])),
        batch_size=batch_size, shuffle=True)

    recon_dataset = datasets.MNIST('data/mnist', train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(), binariser])).test_data[:32]\
                            .float()/255.

    try:
        for epoch in range(1, epochs + 1):
            train(model, epoch, train_loader, optimizer)
            test(model, epoch, test_loader)
            model.reconstruct(recon_dataset, epoch)
            model.sample(epoch)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
    torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
