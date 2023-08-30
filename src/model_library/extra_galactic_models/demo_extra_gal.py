import nifty7 as ift
from ..Model import Model
import libs as Efg


class ExtraGalDemoModel(Model):
    def __init__(self, target_domain, args):
        # This is a completely cooked up extra-galactic RM model for illustrative purposes only.
        # The model is RM_egal = e**(sigma_a * \xi_a + \mu_a) - e**(sigma_b * \xi_b + \mu_b)/(ln(1 + e^z)),
        # where the sigmas and mus are a hyper-parameters of the model,
        # xi_a and  xi_b are fields and z is a number.

        self.chi_lum = args['chi_lum']
        self.chi_red = args['chi_red']
        self.sigma_int_0 = args['sigma_int_0']
        self.sigma_env_0 = args['sigma_env_0']
        self.L = args['L']
        self.z = args['z']


        super().__init__(target_domain)


        #new formula -> Rm^2 = (L/L0)^Xlum * sigma2_int_0/(1+z)^4 + D/D0 * sigma2_env_0
        ## D = integral 0 to z (c/H) * ((1+z)^(4 + Xred)) dz
        ## Xlum, Xred, sigma2_int_0, sigma2_env_0 to be provided in input, looping through values, in order
        ## to calculate different Rm^2, to be applied to Gaussian. Target is eg_contr (e_model)
        ## L0, D0 hyperpars (fixed), c hyperpar (speedlight), H hyperpar (depends on cosmology (refer to already existing code Valentina))
        # input: values from catalog (L, z)
        # output: rm^2, need to calculate eg_contr=G(0, rm^2) as output of function (see numpy.random.normal)
        # all outputs need to be put in some kind of array

    def set_model(self):
        #all parameters are numbers, except constants
        chi_lum = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_lum')
        chi_red = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_red')
        sigma_int_0 = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'sigma_int_0')
        sigma_env_0 = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'sigma_env_0')
        L = ift.FieldAdapter(self.target_domain, 'L')
        z = ift.FieldAdapter(self.target_domain, 'z')

        L0 = Efg.const['L0']
        D0 = Efg.const['D0']

        pass
        # chi_a = ift.FieldAdapter(self.target_domain, 'chi_a')
        # chi_b = ift.FieldAdapter(self.target_domain, 'chi_b')

        # add_mu_a = ift.Adder(ift.full(self.target_domain, self.mu_a))
        # add_mu_b = ift.Adder(ift.full(self.target_domain, self.mu_b))

        # multiply_sigma_a = ift.makeOp(ift.full(self.target_domain, self.sigma_a))
        # multiply_sigma_b = ift.makeOp(ift.full(self.target_domain, self.sigma_b))

        # a = (add_mu_a @ multiply_sigma_a @ chi_a).exp()
        # b = (add_mu_b @ multiply_sigma_b @ chi_b).exp()

        # z = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'z')
        # add_one = ift.Adder(ift.Field(ift.DomainTuple.scalar_domain(), 1.))

        # # the expander matches the scaler domain of z to the domain of a and b
        # expander = ift.VdotOperator(ift.full(self.target_domain, 1.)).adjoint

        # z_log1p = expander @ add_one @ z.exp()
        # self._model = a - b * z_log1p.reciprocal()
        # self._components.update({'a': a, 'b': b, 'z': z, })
