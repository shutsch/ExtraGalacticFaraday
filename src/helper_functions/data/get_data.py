import numpy as np
from src.helper_functions.misc import gal2gal



def get_data(self, data):

    #all data
    rm = np.array(data['rm'])
    rm_err = np.array(data['rm_err'])
    lrm = len(e_rm)
    
    F = np.array(data['stokesI'])

    #extragalactic line of sights
    z_indices = ~np.isnan(data['z_best'])
    
    e_rm = np.array(data['rm'][z_indices])
    e_rm_err = np.array(data['rm_err'][z_indices])
    lerm = len(e_rm)
    
    e_z = np.array(data['z_best'][z_indices])
    e_F = np.array(data['stokesI'][z_indices])

    #galactic line of sights
    g_rm = np.array(data['rm'][~z_indices])
    g_rm_err = np.array(data['rm_err'][~z_indices])
    lgrm=len(g_rm)

    g_F = np.array(data['stokesI'][~z_indices])

    eg_l = np.array(data['l'])
    eg_b = np.array(data['b'])

    theta_eg, phi_eg = gal2gal(eg_l, eg_b) # converting to colatitude and logitude in radians

    ltheta=len(data['theta'])
    lthetaeg = len(theta_eg)

    return {'rm': rm, 'rm_err': rm_err, 'F': F,'lrm': lrm,
        'z_indices': z_indices, 'e_rm': e_rm, 'e_rm_err': e_rm_err, 'e_z': e_z, 'e_F': e_F, 'lerm': lerm,
            'g_rm': g_rm, 'g_rm_err': g_rm_err, 'g_F': g_F, 'lgrm': lgrm, 'eg_l': eg_l, 'eg_b': eg_b,
            'theta_eg': theta_eg, 'phi_eg': phi_eg, 'ltheta': ltheta, 'lthetaeg': lthetaeg    
            }
