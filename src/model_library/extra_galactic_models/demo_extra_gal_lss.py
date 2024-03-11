import nifty8 as ift
from ..Model import Model
import libs as Egf
import numpy as np
import scipy as sp
from astropy.cosmology import FlatLambdaCDM
import math as m
from ...operators.InverseGamma import InverseGammaOperator

class ExtraGalDemoModel(Model):
    def __init__(self, target_domain, args):

        self.L = args['L']
        self.N_los = args['N_los']

        super().__init__(target_domain)

 
    def set_model(self):

        #new formula -> 
        # sigmaRm^2 = !/N_los *Sum_i=1^N_los (L_gc*sigma_gc**2+L_f*sigma_f**2+L_v*sigma_v**2+L_s*sigma_s**2) +sigma_int**2

        #chi_lum = InverseGammaOperator(self.target_domain, self.alpha, self.q) @ ift.FieldAdapter(self.target_domain, 'chi_lum')
      
        chi_cg = ift.FieldAdapter(self.target_domain, 'chi_cg')
        chi_f = ift.FieldAdapter(self.target_domain, 'chi_f')
        chi_v = ift.FieldAdapter(self.target_domain, 'chi_v')
        chi_s = ift.FieldAdapter(self.target_domain, 'chi_s')
        chi_int = ift.FieldAdapter(self.target_domain, 'chi_int')


        L_cg = L[0]
        L_f  = L[1]
        L_v  = L[2]
        L_s  = L[3]

        multiply_L_cg = ift.makeOp(ift.Field(self.target_domain, L_cg),sampling_dtype=float)
        multiply_L_f  = ift.makeOp(ift.Field(self.target_domain, L_f),sampling_dtype=float)
        multiply_L_v  = ift.makeOp(ift.Field(self.target_domain, L_v),sampling_dtype=float)
        multiply_L_s  = ift.makeOp(ift.Field(self.target_domain, L_s),sampling_dtype=float)

        divide_Nlos  = ift.makeOp(ift.Field(self.target_domain, 1/N_los),sampling_dtype=float)


        normalized_L_domain = ift.RGSpace(N_los, 1.) # that's the Nlos domain. The distance between pixels is set to one, as we will manually mutiply with the real z distance later, since it is not the same for each LoS.
        
        full_domain = ift.DomainTuple.make((self.target_domain[0], normalized_L_domain,))
        integrator = ift.ContractionOperator(full_domain, spaces=1) # this is the integration operator, mapping the full domain on the target_domain via a sum
        expander = integrator.adjoint # the adjoint of this operator projects a field in the target_domain onto the full_domain

        sigma_cg2 = multiply_L_cg @ expander @ chi_cg.exp()
        sigma_f2  = multiply_L_f @ expander @ chi_f.exp()
        sigma_v2  = multiply_L_v @ expander @ chi_v.exp()
        sigma_s2  = multiply_L_s @ expander @ chi_s.exp()   


        sigma_eg2 = divide_Nlos @ (integrator @ sigma_cg2 + integrator @ sigma_f2 + integrator @ sigma_v2 + integrator @ sigma_s2) 
      
        sigmaRm2  = sigma_eg2 + chi_int.exp() 
      
        self._model = sigmaRm2
        self._components.update({'sigma_cg^2': chi_cg.exp(), 'sigma_f^2': chi_f.exp(),  'sigma_v^2': chi_v.exp(), 'sigma_f^2': chi_v.exp(),  'sigma_int^2': chi_int.exp()})


