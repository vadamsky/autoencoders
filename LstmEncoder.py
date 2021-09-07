import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime


from BaseAutoEncoder import BaseAutoEncoder, additionalParamsDefault
from AENet import activatorsDict


#Encoder
class Q_net(nn.Module):
    def __init__(self, X_dim=28, N1=32, N2=32,
                 z_dim=6, dropout=0.1, sparse=False, 
                 losstype='binary_cross_entropy',
                 activators=['relu', 'relu', '',
                             'relu', 'relu', 'sigmoid']):
        super(Q_net, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.losstype = losstype
        # torch net
        self.activators = [activatorsDict[a] for a in activators]
        # net
        if self.sparse:
            N2 = N1
        self.lin1 = nn.Linear(X_dim, N1)
        if not self.sparse:
            self.lin2 = nn.Linear(N1, N2) # for not-sparse only
        self.lin3 = nn.Linear(N2, z_dim)
        
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = eval(self.activators[0] + 'x)')
        if not self.sparse:
            x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
            x = eval(self.activators[1] + 'x)')
        x = self.lin3(x)
        x = eval(self.activators[2] + 'x)')
        return x



    
    
    
    
input_size = 4
z_dim = 2

ys = [[[0.0, 0.0, 0.0, 0.0],
       [0.1, 0.15, 0.2, 0.25],
       [0.2, 0.3, 0.4, 0.5],
       [0.3, 0.45, 0.6, 0.75],
       [0.4, 0.6, 0.8, 1.0]],
      [[0.0, 0.0, 0.0, 0.0],
       [0.05, 0.075, 0.1, 0.125],
       [0.1, 0.15, 0.2, 0.25],
       [0.15, 0.225, 0.3, 0.325],
       [0.2, 0.3, 0.4, 0.5]]]

class LSTM(nn.Module):
    def __init__(self, input_dim, z_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim, self.z_dim, self.num_layers)
        self.decoder = nn.LSTM(self.z_dim, self.input_dim, self.num_layers)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        # It is way more general that way
        encoded = last_hidden.repeat((len(input), 1, 1))#input.shape)
        
        # Decode
        y, _ = self.decoder(encoded)
        return torch.squeeze(y)


model = LSTM(input_dim=input_size, z_dim=z_dim, num_layers=1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


for i in range(10000):
    for y in ys:
        y = torch.Tensor(y)
        x = y.view(len(y), -1, input_size)
        #
        y_pred = model(x)
        optimizer.zero_grad()
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            #print(y_pred)
            print(loss)
    if i % 100 == 0:
        print("----------------------")    
    
    
    
    
    
    
    
    
    
    
class LstmEncoder(BaseAutoEncoder):

    def __init__(self, X_dim=28, N1=32, N2=32, z_dim=2, 
                       epochs=500, 
                       dropout=0.1, opt_s='Adam', seed=None,
                       additionalParams=additionalParamsDefault, 
                       real_gauss_A=4.):
        super(AaeEncoder, self).__init__(X_dim, N1, N2, z_dim, 
                                         epochs, dropout, opt_s, seed, "Simple",
                                         additionalParams)
        self.real_gauss_A = real_gauss_A

        self.string_params = "z_%d;N1_%d;N2_%d;opt_%s;dropout=%f;gaussA_%f" % (
            self.z_dim, self.N1, self.N2, self.opt_s, self.dropout, self.real_gauss_A)


    def _report_loss(self, epoch, D_loss_gauss, G_loss, recon_loss):
        print('Epoch-{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch, D_loss_gauss.data,
                                                                                       G_loss.data,
                                                                                       recon_loss.data))
#                                                                                       D_loss_gauss.data[0],
#                                                                                       G_loss.data[0],
#                                                                                       recon_loss.data[0]))

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
            #recon_loss = F.mse_loss(X_sample, X.resize(self.train_batch_size, self.X_dim))
            #recon_loss = F.mse_loss(X_sample, X)
            recon_loss = eval('torch.nn.functional.' + self.losstype + '(X_sample, X)')

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
            #z_real_gauss = Variable(torch.randn(self.train_batch_size, self.z_dim) * self.real_gauss_A)
            z_real_gauss = Variable(torch.randn(len(X), self.z_dim) * self.real_gauss_A)
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

        self.Q = Base_net(X_dim=self.X_dim, N1=self.N1, N2=self.N2, z_dim=self.z_dim, 
                          dropout=self.dropout, sparse=self.sparse, losstype=self.losstype, 
                          activators=self.activators, typeNet="Q")
        self.P = Base_net(X_dim=self.X_dim, N1=self.N1, N2=self.N2, z_dim=self.z_dim, 
                          dropout=self.dropout, sparse=self.sparse, losstype=self.losstype, 
                          activators=self.activators, typeNet="P")
        self.D_gauss = Base_net(N1=self.N1, N2=self.N2, z_dim=self.z_dim, 
                          dropout=self.dropout, sparse=self.sparse, losstype=self.losstype, 
                          activators=self.activators, typeNet="D")
        if self.cuda:
            self.Q = self.Q.cuda()
            self.P = self.P.cuda()
            self.D_gauss = self.D_gauss.cuda()

        # Set learning rates
        gen_lr = 0.0001

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
                losses.append([epoch, D_loss_gauss.data, G_loss.data, recon_loss.data])
                #losses.append([epoch, D_loss_gauss.data[0], G_loss.data[0], recon_loss.data[0]])

        self._save_model(self.Q, 'Q__' + self.string_params + '.pickle')
        self._save_model(self.P, 'P__' + self.string_params + '.pickle')
        self._save_model(self.D_gauss, 'D_gauss__' + self.string_params + '.pickle')
        return self.Q, self.P, self.D_gauss, losses

    def load_model(self, string):
        self.Q = self._load_model('Q__' + string + '.pickle')
        self.P = self._load_model('P__' + string + '.pickle')
        self.D_gauss = self._load_model('D_gauss__' + string + '.pickle')

    # from base
    def _save_model(self, model, filename):
        torch.save(model, filename)


    def get_z(self, data):
        if self.cuda:
            return self.model.cpu().encode(data)
        else:
            return self.model.encode(data)