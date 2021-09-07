import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from AENet import AENet

additionalParamsDefault = {'noise': False,
                           'sparse': False,
                           'losstype': 'binary_cross_entropy', 
                           'activators': ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
                           }

class BaseAutoEncoder(object):
    def __init__(self, X_dim=28, N1=16, N2=8, z_dim=2, 
                 epochs=500, 
                 dropout=0.1, opt_s='Adam', seed=None,
                 aeType='Simple', 
                 additionalParams=additionalParamsDefault):
        #
        self.seed = datetime.datetime.now().microsecond if (seed is None) else seed
        self.X_dim = X_dim
        self.N1 = N1
        self.N2 = N2
        self.z_dim = z_dim

        self.epochs = epochs
        self.dropout = dropout

        self.opts = {'Adam': optim.Adam, 'SGD': optim.SGD}
        self.opt_s = opt_s
        self.opt = self.opts.get(opt_s, optim.Adam)
        if self.opt == optim.Adam:
            self.opt_s = 'Adam'

        self.aeType = aeType
        self.string_params = "%s;seed_%d;z_%d;N1_%d;N2_%d;opt_%s;dropout=%f" % (
            aeType, self.seed, self.z_dim, self.N1, self.N2, self.opt_s, self.dropout)

        self.cuda = torch.cuda.is_available()
        
        self.model = None
        #
        self.sparse = False
        if 'sparse' in additionalParams.keys():
            self.sparse = additionalParams['sparse']
        #
        self.losstype = 'binary_cross_entropy'
        if 'losstype' in additionalParams.keys():
            self.losstype = additionalParams['losstype']
        #
        self.activators = ['relu', 'relu', 'relu',
                           'relu', 'relu', 'sigmoid']
        if 'activators' in additionalParams.keys():
            self.activators = additionalParams['activators']
        #


    def _save_model(self, model, filename):
        torch.save(model, filename)

    def _load_model(self, filename):
        return torch.load(filename)

    def _report_loss(self, epoch, loss):
        print('Epoch-{}; loss: {:.4}'.format(epoch, float(loss)))

    def _train(self, opt, epoch, data):
        '''
        Train procedure for one epoch.
        '''
        return 0

    
    def generate_model(self, data):
        torch.manual_seed(self.seed)

        if self.cuda:
            self.model = AENet(self.aeType, X_dim=self.X_dim, N1=self.N1, N2=self.N2, z_dim=self.z_dim,
                               dropout=self.dropout, sparse=self.sparse, losstype=self.losstype, activators=self.activators).cuda()
        else:
            self.model = AENet(self.aeType, X_dim=self.X_dim, N1=self.N1, N2=self.N2, z_dim=self.z_dim, 
                               dropout=self.dropout, sparse=self.sparse, losstype=self.losstype, activators=self.activators)
        # Set learning rates
        gen_lr = 0.0001

        # Set optimizators
        if self.opt == optim.SGD:
            opt_v = self.opt(self.model.parameters(), lr=gen_lr)
        else:
            opt_v = self.opt(self.model.parameters())

        losses = []
        for epoch in range(self.epochs):
            loss = self._train(opt_v, epoch, data)
            if epoch % 10 == 0:
                self._report_loss(epoch, loss)
                losses.append([epoch, loss])

        self._save_model(self.model, self.aeType + '__' + self.string_params + '.pickle')
        return self.model, losses

    def load_model(self, string):
        self.model = self._load_model(self.aeType + '__' + string + '.pickle')

    def get_z(self, data):
        if self.cuda:
            return self.model.cpu().encode(data)
        else:
            return self.model.encode(data)