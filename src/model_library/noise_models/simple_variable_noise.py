import nifty8 as ift
import numpy as np

from ..Model import Model


class SimpleVariableNoise(Model):

    def __init__(self, target_domain, noise_cov, alpha, q, name=''):
        if isinstance(noise_cov, ift.Field):
            self.noise_cov = ift.makeOp(noise_cov)
        else:
            self.noise_cov = ift.makeOp(ift.Field(self.target_domain, noise_cov))
        self.alpha = alpha
        if isinstance(q, str):
            if q == 'mean':
                q = alpha - 1
            elif q == 'mode':
                q = alpha + 1
            elif q == 'log_mean':
                import scipy.special as sp
                q = np.exp(sp.digamma(alpha))
            else:
                raise ValueError(q, ' is an unknown rule for setting q')
        self.q = q
        self.name = name if name == '' else '_' + name
        super().__init__(target_domain)

    def set_model(self):
        excitations = ift.FieldAdapter(self.target_domain, 'noise_excitations' + self.name)
        from ...operators.InverseGamma import InverseGammaOperator
        eta = InverseGammaOperator(self.target_domain, self.alpha, self.q) @ excitations

        self._model = self.noise_cov @ eta
        self._components = {'eta': eta}
