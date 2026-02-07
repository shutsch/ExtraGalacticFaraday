import numpy as np
from astropy.io import fits



def write_to_file(params, dest_data):


    catalog_name=""
    if params['params_mock_cat.maker_params.surveys.make_survey1']:
        catalog_name=params['file_params.cat_path']+params['params_mock_cat.maker_params.surveys.name1']+'_catalog'+'.fits'
    else:
        catalog_name=params['file_params.cat_path']+params['params_mock_cat.maker_params.base_catalog']+'.fits'
    
    try:
        hdu= fits.open(catalog_name)

        hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = dest_data['rm']
        hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['rm_err']
        hdu[1].data['z_best'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['z_best']
        hdu[1].data['stokesI'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['stokesI']
        new_catalog_name= params['file_params.cat_path']+params['params_mock_cat.maker_params.base_catalog']+'_sim.fits'
        hdu.writeto(new_catalog_name, overwrite=True)
        hdu.close()
        print("Mock catalog written to file successfully.")
        return new_catalog_name
    
    except Exception as e:
        return "Error in writing mock catalog to file: ", e

    




