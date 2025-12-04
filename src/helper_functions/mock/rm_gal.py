import nifty8 as ift
import libs as Egf 
from mock_seb23 import seb23
import utilities as U



def rm_gal(params, sky_domain):

    
    if(params['params_mock_cat.maker_params.maker_type'] == "seb23" or params['params_mock_cat.maker_params.maker_type'] == "ymw16" ):

        rm, B, dm =seb23(params)

        Egf.plot_rmgal(params, dm, B, rm, figname='Mock_cat_Seb23_dm_b_rm.png') 
        rm_gal= rm if params['params_mock_cat.maker_params.disk_on']==1 else\
            B


    if(params['params_mock_cat.maker_params.maker_type'] == "consistent"): #CONSISTENT catalog
        galactic_model = U.get_galactic_model(sky_domain, params)
        gal_mock_position = ift.from_random(galactic_model.get_model().domain, 'normal')
        gal=galactic_model.get_model()(gal_mock_position)
        rm_gal=gal
        Egf.plot_rmgalonly(params, gal, figname='Mock_RM_gal_consistent.png') 

    return rm_gal