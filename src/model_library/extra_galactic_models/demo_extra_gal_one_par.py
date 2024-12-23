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

        self.z = args['z']
        self.F = args['F']

        
        super().__init__(target_domain)

 
    def set_model(self):

        

        multiply_sigma1 = ift.makeOp(ift.full(ift.DomainTuple.scalar_domain(), 0.5), sampling_dtype=float)
        add_mu1 = ift.Adder(ift.full(ift.DomainTuple.scalar_domain(), 4.0))

        chi1 = add_mu1 @ multiply_sigma1 @ ift.FieldAdapter(ift.DomainTuple.scalar_domain(), 'chi1') 


        expander_chi = ift.VdotOperator(ift.full(self.target_domain, 1.)).adjoint

        
        sigmaRm2=(expander_chi @ chi1).exp() 


     
        self._model = sigmaRm2
        self._components.update({'chi1': chi1})

