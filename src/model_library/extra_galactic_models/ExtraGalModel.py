import nifty8 as ift
from ..Model import Model
import libs as Egf
import numpy as np
import scipy as sp
from astropy.cosmology import FlatLambdaCDM
import math as m
from ...operators.InverseGamma import InverseGammaOperator

class ExtraGalModel(Model):
    def __init__(self, target_domain, args, use_prior_params=False):

        self.z = args['z']
        self.F = args['F']
        self.params = args['params']
        self.use_prior_params = use_prior_params
        
        super().__init__(target_domain)

 
    def set_model(self):
        mean_one = self.params['prior_mean.prior_mean_one'] if self.use_prior_params else self.params['mean.mean_one']
        std_one = self.params['prior_std.prior_std_one'] if self.use_prior_params else self.params['std.std_one']
        mean_int = self.params['prior_mean.prior_mean_int'] if self.use_prior_params else self.params['mean.mean_int']
        std_int = self.params['prior_std.prior_std_int'] if self.use_prior_params else self.params['std.std_int']
        mean_env = self.params['prior_mean.prior_mean_env'] if self.use_prior_params else self.params['mean.mean_env']
        std_env = self.params['prior_std.prior_std_env'] if self.use_prior_params else self.params['std.std_env']
        mean_lum = self.params['prior_mean.prior_mean_lum'] if self.use_prior_params else self.params['mean.mean_lum']
        std_lum = self.params['prior_std.prior_std_lum'] if self.use_prior_params else self.params['std.std_lum']
        mean_red = self.params['prior_mean.prior_mean_red'] if self.use_prior_params else self.params['mean.mean_red']
        std_red = self.params['prior_std.prior_std_red'] if self.use_prior_params else self.params['std.std_red']

        multiply_sigma_lum = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_lum), sampling_dtype=float)
        multiply_sigma_int = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_int), sampling_dtype=float)
        multiply_sigma_red = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_red), sampling_dtype=float)
        multiply_sigma_env = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_env), sampling_dtype=float)
        multiply_sigma1 = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), std_one), sampling_dtype=float)

        add_mu_lum = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_lum))
        add_mu_int = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_int))
        add_mu_red = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_red))
        add_mu_env = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_env))
        add_mu1 = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), mean_one))

        chi_env_0 = add_mu_env @ multiply_sigma_env @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_env_0')
        chi_red = add_mu_red @ multiply_sigma_red @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_red') 
        chi_int_0 = add_mu_int @ multiply_sigma_int @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_int_0') 
        chi_lum = add_mu_lum @ multiply_sigma_lum @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi_lum')
        chi1 = add_mu1 @ multiply_sigma1 @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi1') 

        light_speed =  Egf.const['c']
        h =  Egf.const['Jens']['h']
        Wm = Egf.const['Jens']['Wm']
        Wc = Egf.const['Jens']['Wc']
        Wl = Egf.const['Jens']['Wl']
        H0 = 100 * h
            
        cosmo = FlatLambdaCDM(H0=H0, Om0=Wm)  
        L0 = float(Egf.const['L0'])
        D0 = Egf.const['D0']
        Dl=cosmo.luminosity_distance(self.z).value
        factor = float(Egf.const['factor'])

        nz = Egf.const['nz']  # number of redshift bins


        expander_chi = ift.VdotOperator(ift.full(self.target_domain, 1.)).adjoint

        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 1): #1 param


            sigmaRm2=(expander_chi @ chi1).exp() 
            self._model = sigmaRm2
            self._components.update({'chi1': chi1})

        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 2): #2 param

            multiply_z = ift.makeOp(ift.Field(self.target_domain, 1./(1+self.z)**4),sampling_dtype=float)
            multiply_L = ift.makeOp(ift.Field(self.target_domain, np.log(self.F*4*m.pi*Dl**2*factor/L0)),sampling_dtype=float)
            norm=(multiply_L @ expander_chi(chi_lum)).exp()
            term= multiply_z @ expander_chi(chi_int_0.exp())

            fact1 = norm * term

            
            sigmaRm2 = fact1 
            
        
            self._model = sigmaRm2
            self._components.update({'chi_lum': chi_lum, 'chi_int_0': chi_int_0, })

         
           
        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 22): #2 param with redshift included

            normalized_z_domain = ift.RGSpace(nz, 1/nz) # that's the redshift domain. The volume is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
            
            full_domain = ift.DomainTuple.make((self.target_domain[0], normalized_z_domain,))
            integrator = ift.ContractionOperator(full_domain, spaces=1) # this is the integration operator, mapping the full domain on the target_domain via a sum
            expander = integrator.adjoint # the adjoint of this operator projects a field in the target_domain onto the full_domain
            
            # constructing the z_grid field
            z_grid = np.empty(full_domain.shape) 
            for i, z in enumerate(self.z):
                z_grid[i] = np.linspace(1, 1 + z, nz) 
            z_grid = ift.Field(full_domain, z_grid)
        
            # now we proceed as before, just that the operators are defined on the full combined domain
            add_4 = ift.Adder(ift.full(full_domain, 4))
            multiply_1pz = ift.makeOp(z_grid.log(), sampling_dtype=float)
            fact3 = (multiply_1pz @ add_4 @ expander @ expander_chi @ chi_red).exp()  # expander maps chi_red on the full domain
            fact4 = ift.makeOp(((light_speed/(H0*(Wm*z_grid**3+Wc*z_grid**2 +Wl)**0.5))*(1/D0)),sampling_dtype=float)

            fact5 = fact4 @ fact3

            z_weights = ift.makeOp(ift.Field(self.target_domain, self.z / nz),sampling_dtype=float) # these are the z_weights to rescale the integral accordingly
            fact6 = z_weights @ integrator @ (fact5 * (expander @ expander_chi @ chi_env_0.exp()))


            
            sigmaRm2 = fact6
            
        
            self._model = sigmaRm2
            self._components.update({'chi_red': chi_red, 'chi_env_0': chi_env_0, })


        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 3): #3 param
        

            multiply_z = ift.makeOp(ift.Field(self.target_domain, 1./(1+self.z)**4),sampling_dtype=float)
            multiply_L = ift.makeOp(ift.Field(self.target_domain, np.log(self.F*4*m.pi*Dl**2*factor/L0)),sampling_dtype=float)
            norm=(multiply_L @ expander_chi(chi_lum)).exp()
            term= multiply_z @ expander_chi(chi_int_0.exp())

            fact1 = norm * term
        
            fact6 = (expander_chi @ chi_env_0).exp() 

            
            sigmaRm2 =  fact1 + fact6

            self._model = sigmaRm2
            self._components.update({'chi_lum': chi_lum, 'chi_int_0': chi_int_0, 'chi_env_0': chi_env_0, })


        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 33): #3 param
 
        
            fact1 = (expander_chi @ chi_int_0).exp() 



            #nz = Egf.const['nz']  # number of redshift bins
            normalized_z_domain = ift.RGSpace(nz, 1/nz) # that's the redshift domain. The volume is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
            
            full_domain = ift.DomainTuple.make((self.target_domain[0], normalized_z_domain,))
            integrator = ift.ContractionOperator(full_domain, spaces=1) # this is the integration operator, mapping the full domain on the target_domain via a sum
            expander = integrator.adjoint # the adjoint of this operator projects a field in the target_domain onto the full_domain
            
            # constructing the z_grid field
            z_grid = np.empty(full_domain.shape) 
            for i, z in enumerate(self.z):
                z_grid[i] = np.linspace(1, 1 + z, nz) 
            z_grid = ift.Field(full_domain, z_grid)
        
            # now we proceed as before, just that the operators are defined on the full combined domain
            add_4 = ift.Adder(ift.full(full_domain, 4))
            multiply_1pz = ift.makeOp(z_grid.log(), sampling_dtype=float)
            fact3 = (multiply_1pz @ add_4 @ expander @ expander_chi @ chi_red).exp()  # expander maps chi_red on the full domain
            fact4 = ift.makeOp(((light_speed/(H0*(Wm*z_grid**3+Wc*z_grid**2 +Wl)**0.5))*(1/D0)),sampling_dtype=float)

            fact5 = fact4 @ fact3

            z_weights = ift.makeOp(ift.Field(self.target_domain, self.z / nz),sampling_dtype=float) # these are the z_weights to rescale the integral accordingly
            fact6 = z_weights @ integrator @ (fact5 * (expander @ expander_chi @ chi_env_0.exp()))

            
            sigmaRm2 =  fact1 + fact6

            self._model = sigmaRm2
            self._components.update({'chi_int_0': chi_int_0, 'chi_red': chi_red, 'chi_env_0': chi_env_0, })


        if(self.params['params_mock_cat.maker_params.n_eg_params'] == 4): #4 param

            #new formula -> 
            # sigmaRm^2 = (L/L0)^Xlum * sigma_int_0^2/(1+z)^4 + D/D0 * sigma_env_0^2

            #chi_lum = InverseGammaOperator(self.target_domain, self.alpha, self.q) @ ift.FieldAdapter(self.target_domain, 'chi_lum')


            multiply_z = ift.makeOp(ift.Field(self.target_domain, 1./(1+self.z)**4),sampling_dtype=float)
            multiply_L = ift.makeOp(ift.Field(self.target_domain, np.log(self.F*4*m.pi*Dl**2*factor/L0)),sampling_dtype=float)
            norm=(multiply_L @ expander_chi(chi_lum)).exp()
            term= multiply_z @ expander_chi(chi_int_0.exp())

            fact1 = norm * term
        
            #nz = Egf.const['nz']  # number of redshift bins
            normalized_z_domain = ift.RGSpace(nz, 1/nz) # that's the redshift domain. The volume is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
            
            full_domain = ift.DomainTuple.make((self.target_domain[0], normalized_z_domain,))
            integrator = ift.ContractionOperator(full_domain, spaces=1) # this is the integration operator, mapping the full domain on the target_domain via a sum
            expander = integrator.adjoint # the adjoint of this operator projects a field in the target_domain onto the full_domain
            
            # constructing the z_grid field
            z_grid = np.empty(full_domain.shape) 
            for i, z in enumerate(self.z):
                z_grid[i] = np.linspace(1, 1 + z, nz) 
            z_grid = ift.Field(full_domain, z_grid)
        
            # now we proceed as before, just that the operators are defined on the full combined domain
            add_4 = ift.Adder(ift.full(full_domain, 4))
            multiply_1pz = ift.makeOp(z_grid.log(), sampling_dtype=float)
            fact3 = (multiply_1pz @ add_4 @ expander @ expander_chi @ chi_red).exp()  # expander maps chi_red on the full domain
            fact4 = ift.makeOp(((light_speed/(H0*(Wm*z_grid**3+Wc*z_grid**2 +Wl)**0.5))*(1/D0)),sampling_dtype=float)

            fact5 = fact4 @ fact3

            z_weights = ift.makeOp(ift.Field(self.target_domain, self.z / nz),sampling_dtype=float) # these are the z_weights to rescale the integral accordingly
            fact6 = z_weights @ integrator @ (fact5 * (expander @ expander_chi @ chi_env_0.exp()))

            
            sigmaRm2 = fact1 + fact6


            ##sigmaRm2 = fact6
            ##in case we are interested in the RM not only in its sigma, we should output the operator that does the sampling
            ##from a Gaussian with this sigma. This should be possible with the following lines
            ##sigmaRm=sigmaRm2.sqrt()
            ##csi_rm = ift.FieldAdapter(sigmaRm.domain, 'csi_rm')
            ##egal_rm = sigmaRm*csi_rm
            
        
            self._model = sigmaRm2
            self._components.update({'chi_lum': chi_lum, 'chi_red': chi_red, 'chi_int_0': chi_int_0, 'chi_env_0': chi_env_0, })

        

        