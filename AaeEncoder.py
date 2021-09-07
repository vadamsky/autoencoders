import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Encoder
class Q_net(nn.Module):
    def __init__(self, Hlayers=1, dropout=0.1, X_dim=28, N=32, z_dim=6):
        super(Q_net, self).__init__()
        self.Hlayers = Hlayers
        self.dropout = dropout

        self.lin1 = nn.Linear(X_dim, N)
        #self.lin2 = nn.Linear(N, N)] * Hlayers
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        #for i in range(self.Hlayers-1):
        #    x = F.dropout(self.lin2[i](x), p=self.dropout, training=self.training)
        #    x = F.sigmoid(x)
        #x = F.dropout(self.lin2[self.Hlayers-1](x), p=self.dropout, training=self.training)
        #x = F.relu(x)
        x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):
    def __init__(self, Hlayers=1, dropout=0.1, X_dim=28, N=32, z_dim=6):
        super(P_net, self).__init__()
        self.Hlayers = Hlayers
        self.dropout = dropout

        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, Hlayers=1, dropout=0.1, N=32, z_dim=6):
        super(D_net_gauss, self).__init__()
        self.Hlayers = Hlayers
        self.dropout = dropout

        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)

class AaeEncoder(object):

    def __init__(self, X_dim=28, Hlayers=1, N=16, z_dim=6, train_batch_size=32, epochs=500, dropout=0.1, real_gauss_A=4., string1='', opt_s='Adam'):
        self.seed = 10
        self.X_dim = X_dim
        self.N = N
        self.z_dim = z_dim
        self.Hlayers = Hlayers

        self.train_batch_size = train_batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.real_gauss_A = real_gauss_A

        self.opts = {'Adam':optim.Adam, 'SGD':optim.SGD}
        self.opt_s = opt_s
        self.opt = self.opts.get(opt_s, optim.Adam)
        if self.opt==optim.Adam:
            self.opt_s = 'Adam'

        self.string_params = "%s;seed_%d;z_%d;N_%d;batch_%d;opt_%s;dropout=%f;gaussA_%f" % (
            string1, self.seed, self.z_dim, self.N, self.train_batch_size, self.opt_s, self.dropout, self.real_gauss_A)

        self.cuda = torch.cuda.is_available()

    def _save_model(self, model, filename):
        torch.save(model, filename)

    def _load_model(self, filename):
        return torch.load(filename)

    def _report_loss(self, epoch, D_loss_gauss, G_loss, recon_loss):
        print('Epoch-{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch,
                                                                                       D_loss_gauss.data[0],
                                                                                       G_loss.data[0],
                                                                                       recon_loss.data[0]))

    def _train(self, P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, data):
        '''
        Train procedure for one epoch.
        '''
        TINY = 1e-15
        # Set the networks in train mode (apply dropout when needed)
        Q.train()
        P.train()
        D_gauss.train()

        for X in data:
            X = X.float()
            X = Variable(X)
            if self.cuda:
                X = X.cuda()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            z_sample = Q(X)
            X_sample = P(z_sample)
            #recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(self.train_batch_size, self.X_dim) + TINY)
            recon_loss = F.mse_loss(X_sample, X.resize(self.train_batch_size, self.X_dim))

            recon_loss.backward()
            P_decoder.step()
            Q_encoder.step()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Regularization phase
            #######################
            # Discriminator
            Q.eval()
            z_real_gauss = Variable(torch.randn(self.train_batch_size, self.z_dim) * self.real_gauss_A)
            if self.cuda:
                z_real_gauss = z_real_gauss.cuda()

            z_fake_gauss = Q(X)

            D_real_gauss = D_gauss(z_real_gauss)
            D_fake_gauss = D_gauss(z_fake_gauss)

            D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

            D_loss.backward()
            D_gauss_solver.step()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            # Generator
            Q.train()
            z_fake_gauss = Q(X)

            D_fake_gauss = D_gauss(z_fake_gauss)
            G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

            G_loss.backward()
            Q_generator.step()

            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

        return D_loss, G_loss, recon_loss

    def generate_model(self, data):
        torch.manual_seed(self.seed)

        if self.cuda:
            self.Q = Q_net(Hlayers=self.Hlayers, dropout=self.dropout, X_dim=self.X_dim, N=self.N, z_dim=self.z_dim).cuda()
            self.P = P_net(Hlayers=self.Hlayers, dropout=self.dropout, X_dim=self.X_dim, N=self.N, z_dim=self.z_dim).cuda()
            self.D_gauss = D_net_gauss(Hlayers=self.Hlayers, dropout=self.dropout, N=self.N, z_dim=self.z_dim).cuda()
        else:
            self.Q = Q_net(Hlayers=self.Hlayers, dropout=self.dropout, X_dim=self.X_dim, N=self.N, z_dim=self.z_dim)
            self.P = P_net(Hlayers=self.Hlayers, dropout=self.dropout, X_dim=self.X_dim, N=self.N, z_dim=self.z_dim)
            self.D_gauss = D_net_gauss(Hlayers=self.Hlayers, dropout=self.dropout, N=self.N, z_dim=self.z_dim)

        # Set learning rates
        gen_lr = 0.0001
        reg_lr = 0.00005

        # Set optimizators
        if self.opt == optim.SGD:
            P_decoder = self.opt(self.P.parameters(), lr=gen_lr)
            Q_encoder = self.opt(self.Q.parameters(), lr=gen_lr)
            Q_generator = self.opt(self.Q.parameters(), lr=gen_lr)
            D_gauss_solver = self.opt(self.D_gauss.parameters(), lr=gen_lr)
        else:
            P_decoder = self.opt(self.P.parameters())
            Q_encoder = self.opt(self.Q.parameters())
            Q_generator = self.opt(self.Q.parameters())
            D_gauss_solver = self.opt(self.D_gauss.parameters())

        losses = []
        for epoch in range(self.epochs):
            D_loss_gauss, G_loss, recon_loss = self._train(self.P, self.Q, self.D_gauss, P_decoder, Q_encoder,
                                                     Q_generator, D_gauss_solver, data)
            if epoch % 10 == 0:
                self._report_loss(epoch, D_loss_gauss, G_loss, recon_loss)
                losses.append([epoch, D_loss_gauss.data[0], G_loss.data[0], recon_loss.data[0]])

        self._save_model(self.Q, 'Q__' + self.string_params + '.pickle')
        self._save_model(self.P, 'P__' + self.string_params + '.pickle')
        self._save_model(self.D_gauss, 'D_gauss__' + self.string_params + '.pickle')
        return self.Q, self.P, self.D_gauss, losses

    def load_model(self, string):
        self.Q = self._load_model('Q__' + string + '.pickle')
        self.P = self._load_model('P__' + string + '.pickle')
        self.D_gauss = self._load_model('D_gauss__' + string + '.pickle')

    def get_z(self, data):
        return self.Q(data)

    def brake(self):
        """
        Stop the car
        """
        return "Braking"


