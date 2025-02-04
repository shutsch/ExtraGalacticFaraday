import nifty8 as ift
from catalog_maker import CatalogMaker
import libs as Egf
import numpy as np
from settings_maker import Settings_Maker
from src.helper_functions.logger import logger
from src.helper_functions.parameters_maker import Parameters_maker
import utilities as U
import matplotlib
matplotlib.use('Agg')



def run_inference(params, settings_params):
    
    Egf.Minimizer(settings_params['minimizer_params'], settings_params['ecomponents'], params).minimize()

if __name__ == '__main__':
    params = Parameters_maker().yaml_values

    # print a RuntimeWarning  in case of underflows
    np.seterr(all='raise')
    # set seed
    seed = params['params_mock_cat.maker_params.seed']
    ift.random.push_sseq_from_seed(seed)
    settings_params = Settings_Maker(params).run_settings()
    run_inference(params, settings_params)
