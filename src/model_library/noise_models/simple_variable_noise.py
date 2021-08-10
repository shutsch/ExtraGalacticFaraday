import nifty7 as ift
import numpy as np


def build_simple_variable_noise(domain, noise_cov, alpha, q_rule='mode'):
    excitations = ift.FieldAdapter(domain, 'noise_excitations_' + name)
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
    eta = InverseGammaOperator(domain, alpha, q) @ excitations
    return noise_cov @ eta
