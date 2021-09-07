import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

aeTypes = ["Simple", "CAE", "VAE", "AAE", "LSTM"]

activatorsDict = {'': '(',
                  'relu': 'F.relu(',
                  'sigmoid': 'torch.sigmoid(',
                  'tanh': 'nn.Tanh()(',
                  'leakyrelu': 'nn.LeakyReLU(0.25)('}
                  
lossesList = ['binary_cross_entropy', 'cross_entropy', 'kl_div', 'mse_loss']

class AENet(nn.Module):
    def __init__(self, aeType, X_dim=28, N1=16, N2=8,
                 z_dim=6, dropout=0.1, sparse=False, 
                 losstype='binary_cross_entropy',
                 activators=['relu', 'relu', 'relu',
                             'relu', 'relu', 'sigmoid']):
        # Check of input parameters correctness
        if aeType not in aeTypes:
            raise ValueError("aeType not in aeTypes")
        #
        super(AENet, self).__init__()
        #
        # matching
        self.aeType = aeType
        self.X_dim = X_dim
        self.N1 = N1
        self.N2 = N2
        self.z_dim = z_dim
        self.dropout = dropout
        self.sparse = sparse
        self.losstype = losstype
        #
        # torch net
        self.activators = [activatorsDict[a] for a in activators]
        # net
        if self.sparse:
            N2 = N1
        self.fc1  = nn.Linear(X_dim, N1)
        if not self.sparse:
            self.fc2  = nn.Linear(N1, N2) # for not-sparse only
        self.fc3  = nn.Linear(N2, z_dim)
        self.fc32 = nn.Linear(N2, z_dim) # for VAE encoder
        #
        self.fc4  = nn.Linear(z_dim, N2)
        if not self.sparse:
            self.fc2  = nn.Linear(N1, N2) # for not-sparse only
        self.fc5  = nn.Linear(N2, N1) # for not-sparse only
        self.fc6  = nn.Linear(N1, X_dim)
        # 
        # encode function choosing
        if aeType == "VAE":
            self.encode = self.encodeVAE
        #
        # forward function choosing
        if aeType == "VAE":
            self.forward = self.forwardVAE
        #
        # loss function choosing
        if aeType == "CAE":
            self.lossFunction = self.lossFunctionCAE
        if aeType == "VAE":
            self.lossFunction = self.lossFunctionVAE
        #
        
        
    # encode functions
    def encodePart(self, x):
        #x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        h1 = eval(self.activators[0] + 'self.fc1(x))')
        if not self.sparse:
            h2 = eval(self.activators[1] + 'self.fc2(h1))')
        else:
            h2 = h1
        return h2
    
    def encode(self, x):
        h2 = self.encodePart(x)
        h3 = eval(self.activators[2] + 'self.fc3(h2))')
        return h3

    def encodeVAE(self, x):
        h2 = self.encodePart(x)
        h3 = (eval(self.activators[2] + 'self.fc3(h2))'), 
              eval(self.activators[2] + 'self.fc32(h2))'))
        return h3

    # decode functions
    def decode(self, z):
        h4 = eval(self.activators[3] + 'self.fc4(z))')
        if not self.sparse:
            h5 = eval(self.activators[4] + 'self.fc5(h4))')
        else:
            h5 = h4
        h6 = eval(self.activators[5] + 'self.fc6(h5))')
        return h6

    # forward functions
    def forward(self, x):
        hid = self.encode(x)
        rec = self.decode(hid)
        return hid, rec

    # VAE only
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forwardVAE(self, x):
        mu, logvar = self.encodeVAE(x)  # .view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), self.reparameterize(mu, logvar), mu, logvar
    
    # loss functions
    def lossFunction(self, x, recon_x):
        # h, W, lam are ambigous
        recLoss = eval('torch.nn.functional.' + self.losstype + '(recon_x, x, reduction="sum")')
        #print(recon_x, x, mse)
        return (recLoss / len(x))

    # Reconstruction + KL divergence losses summed over all elements and batch
    def lossFunctionCAE(self, x, recon_x, h, W, lam):
        """Compute the Contractive AutoEncoder Loss
        Evalutes the CAE loss, which is composed as the summation of a Mean
        Squared Error and the weighted l2-norm of the Jacobian of the hidden
        units with respect to the inputs.
        See reference below for an in-depth discussion:
          #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
        Args:
            `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
              dimensions of the hidden units and input respectively.
            `x` (Variable): the input to the network, with dims (N_batch x N)
            recons_x (Variable): the reconstruction of the input, with dims
              N_batch x N.
            `h` (Variable): the hidden units of the network, with dims
              batch_size x N_hidden
            `lam` (float): the weight given to the jacobian regulariser term
        Returns:
            Variable: the (scalar) CAE loss
        """
        recLoss = eval('torch.nn.functional.' + self.losstype + '(recon_x, x, reduction="sum")')
        
        #mse = torch.nn.MSELoss(recon_x, x, size_average=False)
        # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
        # opposed to #1
        ###dh = 1 - h * h # if tanh == True
        dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
        # Sum through the input dimension to improve efficiency, as suggested in #1
        #print(W, W.shape)
        w_sum = torch.sum(Variable(W)**2, dim=1)
        # unsqueeze to avoid issues with torch.mv
        #w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
        #print(w_sum, w_sum.shape)
        #print(dh**2)
        w_sum = w_sum.unsqueeze(1)
        #print(w_sum, w_sum.shape)
        contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
        return ((recLoss + contractive_loss.mul_(lam)) / len(x))

    # Reconstruction + KL divergence losses summed over all elements and batch
    def lossFunctionVAE(self, recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = eval('torch.nn.functional.' + self.losstype + '(recon_x, x, reduction="sum")')
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return ((BCE + KLD) / len(x))

