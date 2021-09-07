import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, X_dim=28, N=32, z_dim=6, dropout=0.1):
        super(VAE, self).__init__()

        self.seed = 10
        self.X_dim = X_dim
        self.N = N
        self.z_dim = z_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(X_dim, N)
        self.fc21 = nn.Linear(N, z_dim)
        self.fc22 = nn.Linear(N, z_dim)
        
        self.fc3 = nn.Linear(z_dim, N)
        self.fc4 = nn.Linear(N, X_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # print("encode res: ", len(self.fc21(h1)), len(self.fc22(h1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #eps = torch.randn(std.size())#dtype=std.dtype, layout=std.layout, device=std.device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)  # .view(-1, 784))
        z = self.reparameterize(mu, logvar)
        # print("forward res: ", len(self.decode(z)), len(mu), len(logvar))
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class VAE2(nn.Module):
    def __init__(self, X_dim=28, N=32, z_dim=6, dropout=0.1):
        super(VAE2, self).__init__()

        self.seed = 10
        self.X_dim = X_dim
        self.N = N
        self.z_dim = z_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(X_dim, N)
        self.fc2 = nn.Linear(N, N)
        self.fc31 = nn.Linear(N, z_dim)
        self.fc32 = nn.Linear(N, z_dim)
        
        self.fc4 = nn.Linear(z_dim, N)
        self.fc5 = nn.Linear(N, N)
        self.fc6 = nn.Linear(N, X_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #eps = torch.randn(std.size())#dtype=std.dtype, layout=std.layout, device=std.device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = F.sigmoid(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)  # .view(-1, 784))
        z = self.reparameterize(mu, logvar)
        # print("forward res: ", len(self.decode(z)), len(mu), len(logvar))
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class VaeEncoder(object):
    def __init__(self, X_dim=28, N=32, z_dim=6, train_batch_size=32, epochs=500, dropout=0.1, real_gauss_A=4.,
                 string1='', opt_s='Adam'):
        super(VaeEncoder, self).__init__()

        self.seed = 10
        self.X_dim = X_dim
        self.N = N
        self.z_dim = z_dim

        self.train_batch_size = train_batch_size
        self.epochs = epochs
        self.dropout = dropout

        self.opts = {'Adam': optim.Adam, 'SGD': optim.SGD}
        self.opt_s = opt_s
        self.opt = self.opts.get(opt_s, optim.Adam)
        if self.opt == optim.Adam:
            self.opt_s = 'Adam'

        self.string_params = "%s;seed_%d;z_%d;N_%d;batch_%d;opt_%s;dropout=%f" % (
            string1, self.seed, self.z_dim, self.N, self.train_batch_size, self.opt_s, self.dropout)

        self.cuda = torch.cuda.is_available()

    def _save_model(self, model, filename):
        torch.save(model, filename)

    def _load_model(self, filename):
        return torch.load(filename)

    def _report_loss(self, epoch, loss):
        print('Epoch-{}; loss: {:.4}'.format(epoch, float(loss)))

    def _train(self, VAE_opt, epoch, data):
        '''
        Train procedure for one epoch.
        '''
        TINY = 1e-15
        # Set the network in train mode (apply dropout when needed)
        self.model.train()
        train_loss = 0

        for X in data:
            X = X.float()
            X = Variable(X)
            if self.cuda:
                X = X.cuda()

            # Init gradients
            VAE_opt.zero_grad()

            recon_batch, mu, logvar = self.model(X)  # ret of forward
            loss = self.model.loss_function(recon_batch, X, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            VAE_opt.step()

        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #      epoch, train_loss / len(data)))

        return train_loss

    def generate_model(self, data):
        torch.manual_seed(self.seed)

        if self.cuda:
            self.model = VAE2(X_dim=self.X_dim, N=self.N, z_dim=self.z_dim,
                             dropout=self.dropout).cuda()  # .to(self.device)
            # self.D_gauss = D_net_gauss(Hlayers=self.Hlayers, dropout=self.dropout, N=self.N, z_dim=self.z_dim).cuda()
        else:
            self.model = VAE2(X_dim=self.X_dim, N=self.N, z_dim=self.z_dim, dropout=self.dropout)#.to(self.device)

        # Set learning rates
        gen_lr = 0.0001

        # Set optimizators
        if self.opt == optim.SGD:
            VAE_opt = self.opt(self.model.parameters(), lr=gen_lr)
        else:
            VAE_opt = self.opt(self.model.parameters())

        losses = []
        for epoch in range(self.epochs):
            loss = self._train(VAE_opt, epoch, data)
            if epoch % 10 == 0:
                self._report_loss(epoch, loss)
                losses.append([epoch, loss])

        self._save_model(self.model, 'VAE__' + self.string_params + '.pickle')
        return self.model, losses

    def load_model(self, string):
        self.model = self._load_model('VAE__' + string + '.pickle')

    def get_z(self, data):
        return self.model.forward(data)
