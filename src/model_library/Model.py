import nifty7 as ift


class Model:
    def __init__(self, target_domain, hyperparameters):
        self.target_domain = target_domain
        self._components = dict()
        self._model = None
        self.set_model(hyperparameters)
        if not isinstance(self._model, ift.Operator):
            raise TypeError

    def get_model(self):
        return self._model

    def set_model(self, hyperparameters):
        raise NotImplementedError
