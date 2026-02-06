import os
from pathlib import Path
import shutil
import nifty8 as ift
import libs as Egf
import numpy as np
import utilities as U
import matplotlib
matplotlib.use('Agg')

class Settings_Maker():

    def __init__(self, params):
        self.params = params

    def run_settings(self):
        params= self.params

        dirpath = Path(params['file_params.results_path'])
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(params['file_params.results_path'])
        os.makedirs(params['file_params.results_path'])
        sky_domain = ift.makeDomain(ift.HPSpace(params['params_inference.nside']))

        base_catalog_for_mock= params['params_mock_cat.maker_params.base_catalog'] if params['params_mock_cat.maker_params.use_mock'] else None
        
        catalog_definition = Egf.define_catalog(params,base_catalog=base_catalog_for_mock)
        data= catalog_definition['Data for inference']
        print("Catalog used for inference:", catalog_definition['Catalog path'])    

        # filter
        data_catalog=Egf.get_data(data)
        z_indices=data_catalog['z indices']
        e_rm=data_catalog['e_rm']
        e_rm_err=data_catalog['e_rm_err']
        e_z=data_catalog['e_z']
        e_F=data_catalog['e_F']
        lerm=data_catalog['lerm']
        g_rm=data_catalog['g_rm']
        g_rm_err=data_catalog['g_rm_err']
        lgrm=data_catalog['lgrm']




        #eg and gal domain definition

        gal_data_domain, gal_rm, gal_stddev = Egf.gal_settings(lgrm, g_rm, g_rm_err)
        galactic_model = U.get_galactic_model(sky_domain, params)

        
   
        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
        egal_data_domain, egal_rm, egal_stddev = Egf.egal_settings(lerm, e_rm, e_rm_err)
        egal_model_params = {'z': e_z, 'F': e_F, 'params': params}
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params, use_prior_params=True)

        #if we are not interested in the RM but only in its sigma we can consider the eg sigma as a noise and sum the two here. 
        #we include it here but not in the Variable Noise below because the variable noise include the eta factors and applies only to
        #the Tayolor catalog. Here we are considering the LOFAR catalog. When we will include the correlated eg component, the line 
        #below will include again only the noise. 
        

        noise_params = {
            'egal_var': egal_stddev**2,
            'emodel': emodel.get_model()
        }

        egal_inverse_noise = Egf.EgalAddingNoise(egal_data_domain, noise_params, inverse=True).get_model()


        explicit_likelihood = Egf.get_explicit_likelihood(sky_domain, egal_data_domain, data, z_indices,
                                                        egal_rm, egal_inverse_noise, galactic_model)
        
        implicit_likelihood, implicit_noise, implicit_noise_model = Egf.get_implicit_likelihood(params, sky_domain, gal_data_domain, data, z_indices,\
                                                        gal_rm, gal_stddev, galactic_model)   
    

        # set run parameters and start the inference
        components = galactic_model.get_components()

        ecomponents = emodel.get_components()

        sky_models = {'faraday_sky': galactic_model.get_model(), 'profile': components['log_profile'].exp(),
                    'sign': components['sign']}
        power_models = {'log_profile': components['log_profile_amplitude'], 'sign': components['sign_amplitude']}

    
        #the value that we plot are indeed the values in the position field 
        #scatter_pairs = {'intrinsic': (ecomponents['chi_lum'], ecomponents['chi_int_0']),'environmental': (ecomponents['chi_red'], ecomponents['chi_env_0'])}

        plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
                                        'vmin_mean':'-250', 'vmax_mean':'250', 
                                        'vmin_std':'0', 'vmax_std':'80'},
                        'intrinsic': {'x_label': 'chi_lum', 'y_label': 'sigma_int_0'},
                        'environmental': {'x_label': 'chi_red', 'y_label': 'sigma_env_0'}}
        
        likelihoods={'implicit_likelihood': implicit_likelihood, 'explicit_likelihood': explicit_likelihood}

        minimizer_params = {
            'n_global': params['params_inference.nglobal'],
            'kl_type': 'SampledKLEnergy',
            'plot_path': params['file_params.plot_path'],
            'likelihoods': likelihoods,
            'sky_maps': sky_models,
            'power_spectra': power_models,
            'scatter_pairs': None,
            'plotting_kwargs': plotting_kwargs,
            'sigma_rm': data['rm_err'],
            'gal_pos': ~z_indices,
            #'mock_npi_indices': np.load('mock_npi_indices.npy'),
            'deviation': np.load('deviation.npy'),
            'eta': implicit_noise.get_components()['eta'] if implicit_noise_model != None else None
        }

        return {'minimizer_params': minimizer_params, 'ecomponents': ecomponents, 'emodel': emodel}