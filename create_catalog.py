import nifty8 as ift
import numpy as np
import matplotlib
import libs as Egf
matplotlib.use('TkAgg')
import sys

if __name__ == '__main__':
    n = len(sys.argv)
    np.seterr(all='raise')
    params = Egf.Parameters_maker().get_parsed_params()
        
    Egf.CatalogMaker(params).make_catalog()
