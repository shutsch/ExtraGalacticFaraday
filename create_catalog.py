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
    seed = params['params_mock_cat.maker_params.seed_cat']
    maker_type = params['params_mock_cat.maker_params.maker_type']
    ift.random.push_sseq_from_seed(seed)
    
    CatalogMaker(seed, maker_type, params).make_catalog()
