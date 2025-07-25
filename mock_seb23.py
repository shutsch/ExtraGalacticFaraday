import numpy as np
import nifty8 as ift
from astropy.io import fits
import libs as Egf

def pow_spec(k):
    P0, k0, gamma = [Egf.config['params_mock_cat']['power_spec']['P0'], Egf.config['params_mock_cat']['power_spec']['k0'], Egf.config['params_mock_cat']['power_spec']['gamma']]  
    return P0 / (1. + (k/k0)**(-gamma))

def seb23(params):
    # set seed
    seed = params['params_mock_cat.maker_params.seed']
    ift.random.push_sseq_from_seed(seed)    

    N=Egf.config['params_inference']['nside']  #number of pixels
    A = 0.81    # Constant in front of the RM integral in order to have rad/m^2. In that case B is in muG, n is in cm^-3 and the distance in parsec
    
    if(params['params_mock_cat.maker_params.maker_type'] == "seb23"):
        data = fits.open(params['params_inference.dm_path']+'DM_mean_std_Sebastian.fits')
        dm_new_ar=data[1].data['mean']
    if(params['params_mock_cat.maker_params.maker_type'] == "ymw16" ):
        dm_new_ar=np.load(params['params_inference.dm_path']+'ymw16_dm_map.npy')

    s_space = ift.HPSpace(N)
    s_space_domain = ift.makeDomain(ift.HPSpace(N))
    h_space = s_space.get_default_codomain()

    dm_new_field=ift.Field(s_space_domain, dm_new_ar)



    # Operators
    Bh = ift.create_power_operator(h_space, power_spectrum=pow_spec, sampling_dtype=float)
    R= ift.HarmonicTransformOperator(h_space, target=s_space)



    # this is a random magnetic field with variance 1, please note that the Power Spectrum is different from that of Sebastian
    bh = Bh.draw_sample()
    b = R(bh)

    rm_gal=A*b*dm_new_field


    # Get data
    b_data = b.val
    print('mean=', np.mean(b_data))
    print('var=',np.var(b_data))

   
    return rm_gal, b, dm_new_field

