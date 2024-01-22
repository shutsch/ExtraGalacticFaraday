import nifty8 as ift


class Model:
    def __init__(self, target_domain):
        self.target_domain = target_domain
        self._components = dict()
        self._model = None
        self.set_model()
        if not isinstance(self._model, ift.Operator):
            raise TypeError('self._model is a {} instance, should be ift.Operator'.format(type(self._model)))

    def get_model(self):
        return self._model

    def get_components(self):
        return self._components

    def set_model(self):
        raise NotImplementedError
