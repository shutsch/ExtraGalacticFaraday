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

        self.expander_chi = ift.VdotOperator(ift.full(target_domain, 1.)).adjoint

        self.z = args['z']
        self.F = args['F']
        self.params = args['params']
        self.use_prior_params = use_prior_params

        self.light_speed =  Egf.const['c']

        self.h =  Egf.const['Jens']['h']        
        self.Wm = Egf.const['Jens']['Wm']
        self.Wc = Egf.const['Jens']['Wc']
        self.Wl = Egf.const['Jens']['Wl']
        self.H0 = 100 * self.h

        self.L0 = float(Egf.const['L0'])

        self.D0 = Egf.const['D0']   
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Wm)
        self.Dl=self.cosmo.luminosity_distance(self.z).value  

        self.factor = float(Egf.const['factor'])

        self.spix = Egf.const['spix']

        self.nz = Egf.const['nz']  # number of redshift bins

        super().__init__(target_domain)

 
    def set_model(self):
        mh= Egf.Model_Helper(self.params, self.use_prior_params)
        components = mh.build_model()
        
        chi_env_0 = components['chi_env_0']
        chi_red = components['chi_red'] 
        chi_int_0 = components['chi_int_0']
        chi_lum = components['chi_lum']

        sigma_int_2 = mh.sigma_int_squared(self, components)
        sigma_env_2 = mh.sigma_env_squared(self, components)

        sigmaRm2 = sigma_int_2 + sigma_env_2

        ##sigmaRm2 = fact6
        ##in case we are interested in the RM not only in its sigma, we should output the operator that does the sampling
        ##from a Gaussian with this sigma. This should be possible with the following lines
        ##sigmaRm=sigmaRm2.sqrt()
        ##csi_rm = ift.FieldAdapter(sigmaRm.domain, 'csi_rm')
        ##egal_rm = sigmaRm*csi_rm
    
        #self._model = sigmaRm2
        self._model = sigmaRm2

        self._components.update({'chi_lum': chi_lum, 'chi_red': chi_red, 'chi_int_0': chi_int_0, 'chi_env_0': chi_env_0, })


        

        