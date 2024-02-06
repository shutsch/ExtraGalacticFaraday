import nifty8 as ift
import numpy as np

from ..Model import Model


class EgalAddingNoise(Model):

    def __init__(self, target_domain, noise_params, name=''):
        if isinstance(noise_params['egal_var'], ift.Field):
            self.egal_var = ift.makeOp(
                noise_params['egal_var'], 
                 sampling_dtype=float)
            # self.egal_var = ift.makeOp(noise_params['egal_var'], dom=noise_params['egal_var'].domain, sampling_dtype=float)
        else:
            self.egal_var = ift.makeOp(
                ift.Field(self.target_domain, noise_params['egal_var']), 
                 sampling_dtype=float)
            # self.egal_var = ift.makeOp(ift.Field(self.target_domain, noise_params['egal_var']), dom=noise_params['egal_var'].domain, sampling_dtype=float)

        self.emodel = noise_params['emodel']
        self.name = name if name == '' else '_' + name

        super().__init__(target_domain)

    def set_model(self):
        s2 = self.emodel
       
        self._model = self.egal_var + s2
        
