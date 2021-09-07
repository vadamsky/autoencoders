
from AaeEncoder import AaeEncoder
from LstmEncoder import LstmEncoder
from OtherAutoEncoder import OtherAutoEncoder
from AENet import aeTypes, activatorsDict, lossesList
from BaseAutoEncoder import additionalParamsDefault
from utils import *

class AutoEncoder(object):
    def __init__(self, X_dim=28, N1=16, N2=8, z_dim=2, 
                 epochs=500, 
                 dropout=0.1, opt_s='Adam', seed=None,
                 aeType='Simple', 
                 additionalParams=additionalParamsDefault):
        #
        # Check of input parameters correctness
        if 'noise' not in additionalParams.keys():
            additionalParams['noise'] = additionalParamsDefault['noise']
        if 'sparse' not in additionalParams.keys():
            additionalParams['sparse'] = additionalParamsDefault['sparse']
        if 'losstype' not in additionalParams.keys():
            additionalParams['losstype'] = additionalParamsDefault['losstype']
        if 'activators' not in additionalParams.keys():
            additionalParams['activators'] = additionalParamsDefault['activators']
        #
        if aeType not in aeTypes:
            raise ValueError("aeType not in aeTypes")
        #
        if additionalParams['losstype'] not in lossesList:
            raise ValueError("losstype not in lossesList")
        #
        for activator in additionalParams['activators']:
            if activator not in activatorsDict.keys():
                raise ValueError("activator not in activatorsDict")
        #
        # run
        if aeType == "AAE":
            if 'real_gauss_A' in additionalParams.keys():
                real_gauss_A = additionalParams["real_gauss_A"]
            else:
                raise ValueError("In additionalParams need param real_gauss_A for AAE")
            self.encoder = AaeEncoder(X_dim, N1, N2, z_dim, epochs, 
                                      dropout, opt_s, seed, additionalParams, real_gauss_A)
        else if aeType == "LSTM":
            self.encoder = LstmEncoder(X_dim, N1, N2, z_dim, epochs, 
                                      dropout, opt_s, seed, additionalParams)
        else:
            self.encoder = OtherAutoEncoder(X_dim, N1, N2, z_dim, epochs, 
                                      dropout, opt_s, seed, aeType, additionalParams)
        #
        self.noise = False
        if 'noise' in additionalParams.keys():
            self.noise = additionalParams['noise']


    def _save_model(self, model, filename):
        self.encoder._save_model(model, filename)

    def _load_model(self, filename):
        return self.encoder._load_model(filename)

    def generate_model(self, data):
        if self.noise:
            data = noise_input(data)
        return self.encoder.generate_model(data)

    def load_model(self, string):
        self.encoder.load_model(string)

    def get_z(self, data):
        return self.encoder.get_z(data)