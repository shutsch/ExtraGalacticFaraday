import nifty7 as ift
import numpy as np

from ..Model import Model


class SimpleVariableNoise(Model):

    def __init__(self, target_domain, hyperparameters, name=''):
        super().__init__(target_domain, hyperparameters)
        self.name = name

    def set_model(self, hyperparameters):
        excitations = ift.FieldAdapter(self.target_domain, 'noise_excitations_' + self.name)
        alpha = hyperparameters['alpha']
        q_rule = hyperparameters['q_rule']
        noise_cov = hyperparameters['noise_cov']
        if isinstance(noise_cov, ift.Field):
            noise_cov = ift.makeOp(noise_cov)
        else:
            noise_cov = ift.makeOp(ift.Field(self.target_domain, noise_cov))
        if q_rule == 'mean':
            q = alpha - 1
        elif q_rule == 'mode':
            q = alpha + 1
        elif q_rule == 'log_mean':
            import scipy.special as sp
            q = np.exp(sp.digamma(alpha))
        else:
            ValueError(q_rule, ' is an unknown rule for setting q')
        from ...operators.InverseGamma import InverseGammaOperator
        eta = InverseGammaOperator(self.target_domain, alpha, q) @ excitations

        self._model = noise_cov @ eta
