import nifty8 as ift

from ..Model import Model


class StaticNoise(Model):

    def __init__(self, target_domain, noise_cov, inverse):
        self.inverse = inverse
        if not isinstance(noise_cov, ift.Field):
            noise_cov = ift.Field(self.target_domain, noise_cov)
        self.noise_cov = noise_cov
        super().__init__(target_domain)

    def set_model(self):
        if self.inverse:
            self._model = ift.makeOp(self.noise_cov**(-1))
        self._model = ift.makeOp(self.noise_cov)
