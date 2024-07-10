import numpy as np
import nifty8 as ift
from astropy.io import fits

def pow_spec(k):
    #P0, k0, gamma = [1.1e14, 34, -2.4]  # parameters of the power spectrum as defined in Hustschenreuter et al. 2022
    #P0, k0, gamma = [0.1, 34, -2.4]  # test parameters
    #P0, k0, gamma = [4., 2.8, -2.4]  # modified parameters of the power spectrum, more similar to Oppermann (used for the mock catalog)
    P0, k0, gamma = [1.5, 5.0, -2.5]  # modified parameters of the power spectrum Oppermann et al. 2012 (I could not find the results from NO2015)
    return P0 / (1. + (k/k0)**(-gamma))

def seb23(num_seed):
    # set seed
    seed = num_seed 
    ift.random.push_sseq_from_seed(seed)    

    N=256  #number of pixels
    A = 0.81    # Constant in front of the RM integral in order to have rad/m^2. In that case B is in muG, n is in cm^-3 and the distance in parsec
    
    data = fits.open('DM_mean_std_Sebastian.fits')

    dm_new_ar=data[1].data['mean']

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

