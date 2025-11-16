import numpy as np



def sampling_from_distribiution(params, e_z, data, dest_data):

    if params['params_mock_cat.maker_params.surveys.make_survey1'] == True:
        los=int(params['params_mock_cat.maker_params.multiple']*params['params_mock_cat.maker_params.surveys.los1'])
        b_sel_indices=np.where(abs(dest_data['b'][np.where(dest_data['catalog']==params['params_mock_cat.maker_params.surveys.name1'])[0]])>params['params_mock_cat.maker_params.gal_lat_th'])[0] 
    else:
        los=int(params['params_mock_cat.maker_params.multiple']*dest_data['b'].size)
        b_sel_indices=np.where(abs(dest_data['b'])>params['params_mock_cat.maker_params.gal_lat_th'])[0] 

    #creation of indices of mock z
    z_mock_indices=np.unique(np.random.choice(b_sel_indices, size=los))
    lmock=len(z_mock_indices)
    print('Number of LOS with redshift', lmock)
    print('Total number of LOS in the catalog', dest_data['b'].size)


    if params['params_mock_cat.maker_params.nvss']:
        print('Using NVSS redshifts...')

        e_z=np.load(params['file_params.auxiliary_path']+'z_nvss.npy')

        nvss_index=np.where(data['catalog']=="2009ApJ...702.1230T")[0]
        e_F = np.array(data['stokesI'][nvss_index])
    

    return {'F_mock': np.random.choice(e_F[np.where(e_F>0)],size=lmock) ,
            'z_mock': np.random.choice(e_z,size=lmock),
             'z_mock_indices': z_mock_indices }



def sampling_from_noise_distribiution(sigma, lcat):

    return {'sigma': np.random.choice(sigma,size=lcat),
            }

