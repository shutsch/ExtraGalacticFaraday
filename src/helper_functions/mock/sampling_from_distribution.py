import numpy as np



def sampling_from_distribiution(params, e_z, data, lmock):

    if params['params_mock_cat.maker_params.nvss']:
        print('Using NVSS redshifts...')

        e_z=np.load(params['file_params.auxiliary_path']+'z_nvss.npy')

        nvss_index=np.where(data['catalog']=="2009ApJ...702.1230T")[0]
        e_F = np.array(data['stokesI'][nvss_index])
    

    return {'F_mock': np.random.choice(e_F[np.where(e_F>0)],size=lmock) ,
            'z_mock': np.random.choice(e_z,size=lmock) }



def sampling_from_noise_distribiution(sigma, lcat):

    return {'sigma': np.random.choice(sigma,size=lcat),
            }

