import nifty8 as ift
import numpy as np
import matplotlib
import libs as Egf

from catalog_maker import CatalogMaker
matplotlib.use('TkAgg')
import sys

if __name__ == '__main__':
    n = len(sys.argv)
    np.seterr(all='raise')
    seed = Egf.config['params_mock_cat']['maker_params']['seed']
    maker_type = Egf.config['params_mock_cat']['maker_params']['maker_type']
    ift.random.push_sseq_from_seed(seed)
    
    CatalogMaker(seed, maker_type).make_catalog()
