import nifty8 as ift
import libs as Egf 
import numpy as np
import random
import matplotlib.pyplot as plt

def npi(params, dest_data, rm_data, eg_b,eg_gal_data, sigma_mock):
    np.random.seed(seed=params['params_inference.seed'])

    
    delta_rm_list=[]
    if params['params_mock_cat.maker_params.npi.use_npi']==True:
        b_indices=np.where(np.isnan(dest_data['z_best']))[0]
        npi_indices=np.unique(np.random.choice(b_indices, size=params['params_mock_cat.maker_params.npi.npi_los']))
        np.save('mock_npi_indices.npy', npi_indices)
        for item in b_indices:
            if item in npi_indices:
                mu_nvss=params['params_mock_cat.maker_params.npi.mu_nvss']
                sigma_nvss=params['params_mock_cat.maker_params.npi.sigma_nvss']
                delta_rm=np.random.normal(mu_nvss, sigma_nvss)
                if random.choice('+-')=='-':
                    rm_data[item] -= delta_rm
                    delta_rm_list.append(-delta_rm)
                else:
                    rm_data[item] += delta_rm
                    delta_rm_list.append(delta_rm)
        
        delta_rm_array=np.array(delta_rm_list)
        plt.scatter(eg_b[npi_indices], delta_rm_array)
        plt.savefig('Delta_rm.png', bbox_inches='tight')

    deviation=(rm_data-eg_gal_data.val)/sigma_mock
    np.save('deviation.npy',deviation)
    print('Deviation', deviation.size)


    return rm_data