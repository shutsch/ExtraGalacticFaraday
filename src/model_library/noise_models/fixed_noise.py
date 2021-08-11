import nifty7 as ift

from ..Model import Model


class StaticNoise(Model):

    def __init__(self, target_domain, hyperparameters, inverse):
        super().__init__(target_domain, hyperparameters)
        self.inverse = inverse

    def set_model(self, hyperparameters):
        noise_cov = hyperparameters['noise_cov']
        if self.inverse:
            noise_cov = noise_cov ** -1
        if isinstance(noise_cov, ift.Field):
            self._model = ift.makeOp(noise_cov)
        self._model = ift.makeOp(ift.Field(self.target_domain, noise_cov))
