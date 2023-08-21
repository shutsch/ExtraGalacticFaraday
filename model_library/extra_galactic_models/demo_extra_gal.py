import nifty7 as ift
from ..Model import Model


class ExtraGalDemoModel(Model):
    def __init__(self, target_domain, sigma_a, sigma_b, mu_a, mu_b):
        # This is a completely cooked up extra-galactic RM model for illustrative purposes only.
        # The model is RM_egal = e**(sigma_a * \xi_a + \mu_a) - e**(sigma_b * \xi_b + \mu_b)/(ln(1 + e^z)),
        # where the sigmas and mus are a hyper-parameters of the model,
        # xi_a and  xi_b are fields and z is a number.

        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.mu_a = mu_a
        self.mu_b = mu_b
        super().__init__(target_domain)

    def set_model(self):
        chi_a = ift.FieldAdapter(self.target_domain, 'chi_a')
        chi_b = ift.FieldAdapter(self.target_domain, 'chi_b')

        add_mu_a = ift.Adder(ift.full(self.target_domain, self.mu_a))
        add_mu_b = ift.Adder(ift.full(self.target_domain, self.mu_b))

        multiply_sigma_a = ift.makeOp(ift.full(self.target_domain, self.sigma_a))
        multiply_sigma_b = ift.makeOp(ift.full(self.target_domain, self.sigma_b))

        a = (add_mu_a @ multiply_sigma_a @ chi_a).exp()
        b = (add_mu_b @ multiply_sigma_b @ chi_b).exp()

        z = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'z')
        add_one = ift.Adder(ift.Field(ift.DomainTuple.scalar_domain(), 1.))

        # the expander matches the scaler domain of z to the domain of a and b
        expander = ift.VdotOperator(ift.full(self.target_domain, 1.)).adjoint

        z_log1p = expander @ add_one @ z.exp()
        self._model = a - b * z_log1p.reciprocal()
        self._components.update({'a': a, 'b': b, 'z': z, })
