import nifty8 as ift
import numpy as np
import healpy as hp

class RmOperator(ift.LinearOperator):
       def __init__(self, domain, target, rm):
            self._domain = ift.makeDomain(domain)
            self._target = ift.makeDomain(target)
        
            self._capability = self.TIMES | self.ADJOINT_TIMES