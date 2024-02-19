import nifty8 as ift
from ..Model import Model
import libs as Egf
import numpy as np
import scipy as sp
from astropy.cosmology import FlatLambdaCDM
import math as m

class ExtraGalDemoModel(Model):
    def __init__(self, target_domain, args):

        self.z = args['z']
        self.L = args['L']

        super().__init__(target_domain)

 
    def set_model(self):

        #new formula -> 
        # sigmaRm^2 = (L/L0)^Xlum * sigma_int_0^2/(1+z)^4 + D/D0 * sigma_env_0^2
      
        chi_lum = ift.FieldAdapter(self.target_domain, 'chi_lum')
        chi_red = ift.FieldAdapter(self.target_domain, 'chi_red')
        chi_int_0 = ift.FieldAdapter(self.target_domain, 'chi_int_0')
        chi_env_0 = ift.FieldAdapter(self.target_domain, 'chi_env_0')

   
        light_speed =  Egf.const['c']
        h =  Egf.const['Planck']['h']
        Wm = Egf.const['Planck']['Wm']
        Wc = Egf.const['Planck']['Wc']
        Wl = Egf.const['Planck']['Wl']
        H0 = 100 * h
        
        cosmo = FlatLambdaCDM(H0=H0, Om0=Wm)  
        L0 = float(Egf.const['L0'])
        D0 = Egf.const['D0']
        Dl=cosmo.luminosity_distance(self.z)


        multiply_z = ift.makeOp(ift.Field(self.target_domain, 1./(1+self.z)**4),sampling_dtype=float)

        #multiply_L = ift.makeOp(ift.Field(self.target_domain, np.log(self.L/L0)),sampling_dtype=float)
        multiply_L = ift.makeOp(ift.Field(self.target_domain, np.log(self.L*4*m.pi*Dl**2/L0)),sampling_dtype=float)

        norm=(multiply_L @ chi_lum).exp()

        term= multiply_z @ chi_int_0.exp()
       
        fact1 = norm * term



       
        nz = 100  # number of redshift bins
        normalized_z_domain = ift.RGSpace(100, 1.) # that's the redshift domain. The distance between pixels is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
        
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
        fact3 = (multiply_1pz @ add_4 @ expander @ chi_red).exp()  # expander maps chi_red on the full domain
        #fact4 = ift.makeOp((3./(H0*(Wm*z_grid**3+Wc*z_grid**2 +Wl)**0.5)*1/D0),sampling_dtype=float)
        fact4 = ift.makeOp((light_speed/(H0*(Wm*z_grid**3+Wc*z_grid**2 +Wl)**0.5)*1/D0),sampling_dtype=float)

        fact5 = fact4 @ fact3

        z_weights = ift.makeOp(ift.Field(self.target_domain, self.z / nz),sampling_dtype=float) # these are the z_weights to rescale the integral accordingly
        fact6 = z_weights @ integrator @ (fact5 * (expander @ chi_env_0.exp()))

        
        sigmaRm2 = fact1 + fact6
        #sigmaRm2 = fact6
        #in case we are interested in the RM not only in its sigma, we should output the operator that does the sampling
        #from a Gaussian with this sigma. This should be possible with the following lines
        #sigmaRm=sigmaRm2.sqrt()
        #csi_rm = ift.FieldAdapter(sigmaRm.domain, 'csi_rm')
        #egal_rm = sigmaRm*csi_rm
        
      
        self._model = sigmaRm2
        self._components.update({'chi_lum': chi_lum, 'chi_red': chi_red, 'chi_int_0': chi_int_0, 'chi_env_0': chi_env_0,'sigma_int_0': chi_int_0.exp(), 'sigma_env_0': chi_env_0.exp(), })


