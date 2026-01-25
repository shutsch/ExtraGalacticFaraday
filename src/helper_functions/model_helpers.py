import nifty8 as ift
import numpy as np
import math as m


class Model_Helper():
    def __init__(self, params, use_prior_params=False):
        self.params = params 
        self.use_prior_params = use_prior_params   

    def build_model(self):
        mean_int = self.params['prior_parameters.prior_mean.prior_mean_int'] if self.use_prior_params else self.params['mock_cat_initial_position.mean.mean_int']
        std_int = self.params['prior_parameters.prior_std.prior_std_int'] if self.use_prior_params else self.params['mock_cat_initial_position.std.std_int']
        mean_env = self.params['prior_parameters.prior_mean.prior_mean_env'] if self.use_prior_params else self.params['mock_cat_initial_position.mean.mean_env']
        std_env = self.params['prior_parameters.prior_std.prior_std_env'] if self.use_prior_params else self.params['mock_cat_initial_position.std.std_env']
        mean_lum = self.params['prior_parameters.prior_mean.prior_mean_lum'] if self.use_prior_params else self.params['mock_cat_initial_position.mean.mean_lum']
        std_lum = self.params['prior_parameters.prior_std.prior_std_lum'] if self.use_prior_params else self.params['mock_cat_initial_position.std.std_lum']
        mean_red = self.params['prior_parameters.prior_mean.prior_mean_red'] if self.use_prior_params else self.params['mock_cat_initial_position.mean.mean_red']
        std_red = self.params['prior_parameters.prior_std.prior_std_red'] if self.use_prior_params else self.params['mock_cat_initial_position.std.std_red']

        multiply_sigma_lum = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_lum), sampling_dtype=float)
        multiply_sigma_int = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_int), sampling_dtype=float)
        multiply_sigma_red = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_red), sampling_dtype=float)
        multiply_sigma_env = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_env), sampling_dtype=float)

        add_mu_lum = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_lum))
        add_mu_int = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_int))
        add_mu_red = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_red))
        add_mu_env = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_env))

        chi_env_0 = add_mu_env @ multiply_sigma_env @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_env_0')
        chi_red = add_mu_red @ multiply_sigma_red @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_red') 
        chi_int_0 = add_mu_int @ multiply_sigma_int @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_int_0') 
        chi_lum = add_mu_lum @ multiply_sigma_lum @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_lum')

        return {'chi_env_0': chi_env_0, 'chi_red': chi_red,
                'chi_int_0': chi_int_0, 'chi_lum': chi_lum}  
    
    def sigma_int_squared(self, model, components):

        chi_int_0 = components['chi_int_0']
        chi_lum = components['chi_lum']

        z_factor = ift.makeOp(ift.Field(model.target_domain, 1./(1+model.z)**4),sampling_dtype=float)
        lum_exponent = ift.makeOp(ift.Field(model.target_domain, np.log(model.F*4*m.pi*model.Dl**2*model.factor*(1+model.z)**(model.spix-1)/model.L0)),sampling_dtype=float)
        lum_factor =(lum_exponent @ model.expander_chi(chi_lum)).exp()
        sigma_int0_2= z_factor @ model.expander_chi(chi_int_0.exp())
        sigma_int_2 = lum_factor * sigma_int0_2

        return sigma_int_2 
    
    def sigma_env_squared(self, model, components):

        chi_env_0 = components['chi_env_0']
        chi_red = components['chi_red'] 
        
        #nz = Egf.const['nz']  # number of redshift bins
        normalized_z_domain = ift.RGSpace(model.nz, 1/model.nz) #     that's the redshift domain. The volume is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
        
        full_domain = ift.DomainTuple.make((model.target_domain[0], normalized_z_domain,))
        integrator = ift.ContractionOperator(full_domain, spaces=1) # this is the integration operator, mapping the full domain on the target_domain via a sum
        expander_integrator = integrator.adjoint # the adjoint of this operator projects a field in    the target_domain onto the full_domain      
        
        # constructing the z_grid field
        z_grid = np.empty(full_domain.shape) 
        for i, z in enumerate(model.z):
            z_grid[i] = np.linspace(1, 1 + z, model.nz) 
        z_grid = ift.Field(full_domain, z_grid)
    
        # now we proceed as before, just that the operators are defined on the full combined domain
        add_4 = ift.Adder(ift.full(full_domain, 4))
        onepz_factor = ift.makeOp(z_grid.log(), sampling_dtype=float)
        chired_z_factor = (onepz_factor @ add_4 @ expander_integrator @ model.expander_chi @ chi_red).exp()  # expander maps chi_red on the full domain
        integrand_constants = ift.makeOp(((model.light_speed/(model.H0*(model.Wm*z_grid**3+model.Wc*z_grid**2 +model.Wl)**0.5))*(1/model.D0)),sampling_dtype=float)
       
        integrand = integrand_constants @ chired_z_factor
       
        z_weights = ift.makeOp(ift.Field(model.target_domain, model.z / model.nz),sampling_dtype=float) # these are the z_weights to rescale the integral accordingly
       
        sigma_env_2 = z_weights @ integrator @ (integrand * (expander_integrator @ model.expander_chi @ chi_env_0.exp()))


        return sigma_env_2 