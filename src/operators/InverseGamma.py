import numpy as np
import nifty8 as ift
from scipy.stats import invgamma, norm
from scipy.special import gammaincinv, gammainccinv


class InverseGammaOperator(ift.Operator):
    """Transforms a Gaussian into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^(-\\alpha -1)`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto a inverse gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    alpha : float
        The alpha-parameter of the inverse-gamma distribution.
    q : float
        The q-parameter of the inverse-gamma distribution.
    delta : float
        distance between sampling points for linear interpolation.
    """
    def __init__(self, domain, alpha, q):
        self._domain = self._target = ift.DomainTuple.make(domain)
        self._alpha, self._q, = np.float64(alpha), np.float64(q)
        self._xmin, self._xmax = -20., 20.
        # Precompute
        if isinstance(self._alpha, np.ndarray):
            self._delta = np.float64(0.1)
            xs = np.arange(self._xmin, self._xmax + 2 * self._delta, self._delta)

            self._table = np.zeros((domain.shape[0], len(xs), ))
            self._deriv = np.zeros((domain.shape[0], len(xs) - 1, ))
            for i in range(len(self._alpha)):
                self._table[i, :] = np.log(self._q[i]/np.append(gammainccinv(self._alpha[i], norm.cdf(xs[xs <= 0])),
                                                       gammaincinv(self._alpha[i], norm.cdf(-xs[xs > 0]))))
                self._deriv[i, :] = (self._table[i][1:] - self._table[i][:-1]) / self._delta
        else:
            self._delta = np.float64(1e-3)
            xs = np.arange(self._xmin, self._xmax + 2 * self._delta, self._delta)
            self._table = np.log(self._q/np.append(gammainccinv(self._alpha, norm.cdf(xs[xs <= 0])),
                                 gammaincinv(self._alpha, norm.cdf(-xs[xs > 0]))))[:]
            self._deriv = ((self._table[1:]-self._table[:-1]) / self._delta)[:]

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        #tay = isinstance(x, ift.Taylor)
        val = x.val.val if lin else x.val
        val = (np.clip(val, self._xmin, self._xmax) - self._xmin) / self._delta
        if isinstance(self._alpha, np.ndarray):
            d = np.arange(len(self._alpha))
            # Operator
            fi = (np.floor(val).astype(int)).clip(0, len(self._table[0, :] - 1))
            w = val - fi
            res = np.exp((1 - w) * self._table[d, fi] + w * self._table[d, fi + 1])

            points = ift.Field(self._domain, res)
            if not lin:  # ( or tay):
                return points
            if lin:
                # Derivative of linear interpolation
                der = self._deriv[d, fi] * res

                jac = ift.makeOp(ift.Field(self._domain, der))
                jac = jac(x.jac)
                return x.new(points, jac)
        else:
            # Operator
            fi = (np.floor(val).astype(int)).clip(0, len(self._table[:] - 1))
            w = val - fi
            res = np.exp((1 - w) * self._table[fi] + w * self._table[fi + 1])

            points = ift.Field(self._domain, res)
            if not lin:  # ( or tay):
                return points
            if lin:
                # Derivative of linear interpolation
                der = self._deriv[fi] * res

                jac = ift.makeOp(ift.Field(self._domain, der))
                jac = jac(x.jac)
                return x.new(points, jac)

    @staticmethod
    def IG(field, alpha, q):
        foo = invgamma.ppf(norm.cdf(field.val), alpha, scale=q)
        return ift.Field(field.domain, foo)

    @staticmethod
    def inverseIG(u, alpha, q):
        res = norm.ppf(invgamma.cdf(u.val, alpha, scale=q))
        return ift.Field(u.domain, res)

    @property
    def alpha(self):
        return self._alpha

    @property
    def q(self):
        return self._q
