import nifty7 as ift
from ..Model import Model


class ExtraGalDemoModel(Model):
    def __init__(self, target_domain, **hyperparameters):
        # This is a completely cooked up extra-galactic RM model for illustrative purposes only.
        # The model is RM_egal = e**(sigma_a * \xi_a + \mu_a) - e**(sigma_b * \xi_b + \mu_b)/(ln(1 + e^z)),
        # where the sigmas and mus are a hyper-parameters of the model,
        # xi_a and  xi_b are fields and z is a number.

        super().__init__(target_domain, hyperparameters)

    def set_model(self, hyperparameters):
        sigma_a = hyperparameters['sigma_a']
        sigma_b = hyperparameters['sigma_b']
        mu_a = hyperparameters['mu_a']
        mu_b = hyperparameters['mu_b']

        chi_a = ift.FieldAdapter(self.target_domain, 'chi_a')
        chi_b = ift.FieldAdapter(self.target_domain, 'chi_b')

        a = (sigma_a * chi_a + mu_a).exp()
        b = (sigma_b * chi_b + mu_b).exp()

        z = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'z')

        # the expander matches the scaler domain of z to the domain of a and b
        expander = ift.VdotOperator(ift.full(self.target_domain, 1.)).adjoint

        z_log1p = expander @ z.log1p()

        self._model = a - b / z_log1p
        self._components.update({'a': a, 'b': b, 'z': z, })
