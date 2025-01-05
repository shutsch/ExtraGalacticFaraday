import nifty8 as ift
import numpy as np
import matplotlib
import libs as Egf

from catalog_maker import CatalogMaker
from src.helper_functions.parameters_maker import Parameters_maker
matplotlib.use('TkAgg')
import sys

if __name__ == '__main__':
    n = len(sys.argv)
    np.seterr(all='raise')
    params = Parameters_maker().get_parsed_params()
        
    CatalogMaker(params).make_catalog()
