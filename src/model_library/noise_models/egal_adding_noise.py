import nifty8 as ift
import numpy as np

from ..Model import Model


class EgalAddingNoise(Model):

    def __init__(self, target_domain, noise_params, name=''):
        self.egal_var = noise_params['egal_var']
        self.emodel = noise_params['emodel']

        self.name = name if name == '' else '_' + name

        super().__init__(target_domain)

    def set_model(self):
        egal_op = ift.Adder(self.egal_var)
        self._model = egal_op @ self.emodel
        
