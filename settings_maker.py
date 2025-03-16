import nifty8 as ift
from catalog_maker import CatalogMaker
import libs as Egf
import numpy as np
from src.helper_functions.logger import logger
import utilities as U
import matplotlib
matplotlib.use('Agg')

class Settings_Maker():

    def __init__(self, params):
        self.params = params

    def run_settings(self):
        params= self.params

        sky_domain = ift.makeDomain(ift.HPSpace(params['params_inference.nside']))
        catalog_version = 'custom'

        data = Egf.get_rm(filter_pulsars=True, version=f'{catalog_version}', default_error_level=0.5, params=params)

        #create mock catalog option
        if(params['params_mock_cat.maker_params.use_mock']):
            CatalogMaker(params, base_catalog=data).make_catalog()
            data = Egf.get_rm(filter_pulsars=True, version=f'{catalog_version}_sim', default_error_level=0.5, params=params)
            logger.info("CREATED NEW MOCK CATALOG")        

        # filter
        z_indices = ~np.isnan(data['z_best'])

        e_rm = np.array(data['rm'][z_indices])
        e_stddev = np.array(data['rm_err'][z_indices])
        e_z = np.array(data['z_best'][z_indices])
        e_F = np.array(data['stokesI'][z_indices])

        galactic_model = U.get_galactic_model(sky_domain, params)

        egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(e_rm),)))

        egal_rm = ift.Field(egal_data_domain, e_rm)
        egal_stddev = ift.Field(egal_data_domain, e_stddev)
        
        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
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


        
        explicit_response = Egf.SkyProjector(theta=data['theta'][z_indices], phi=data['phi'][z_indices],
                                            domain=sky_domain, target=egal_data_domain) 

        
        #if we are not interested in the RM but only in its sigma we do not need to include the Rm in the following line
        explicit_model = explicit_response @ galactic_model.get_model()
        #egal_model = explicit_response @ galactic_model.get_model() + emodel.get_model()
        residual = ift.Adder(-egal_rm) @ explicit_model
        
        new_dom = ift.MultiDomain.make({'icov': egal_inverse_noise.target, 'residual': residual.target})
        n_res = ift.FieldAdapter(new_dom, 'icov')(egal_inverse_noise) + \
            ift.FieldAdapter(new_dom, 'residual')(residual)
        
        #we need to use the VariableCovarianceGaussianEnerg instead than the GaussianEnergy because the variance (that now
        #includes the eg part that now we are fitting) is varying, is not anymore a costant. When we will include the 
        #correlated eg component we will need to use again the GaussianEnergy. 
        explicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=egal_data_domain, residual_key='residual',
                                                                inverse_covariance_key='icov',
                                                                sampling_dtype=np.dtype(np.float64)) @ n_res
        

        g_rm = np.array(data['rm'][~z_indices])
        g_stddev = np.array(data['rm_err'][~z_indices])

        gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(g_rm),)))

        gal_rm = ift.Field(gal_data_domain, g_rm)
        gal_stddev = ift.Field(gal_data_domain, g_stddev)

        implicit_response = Egf.SkyProjector(theta=data['theta'][~z_indices],
                                            phi=data['phi'][~z_indices],
                                            domain=sky_domain, target=gal_data_domain)


        #to use when inference on the noise factors is necessary
        if self.params['params_mock_cat.maker_params.npi']==1:
        # Possible all sky variation of alpha, requires pygedm package 
            alpha = 2.5
            log_ymw = np.log(Egf.load_ymw_sky('./data/', nside=params['params_inference.nside'], model='ymw16', mode='mc'))
            log_ymw /= log_ymw.min()
            log_ymw *= 5
            alpha = implicit_response(ift.Field(ift.makeDomain(implicit_response.domain), log_ymw)).val

            implicit_noise = Egf.SimpleVariableNoise(gal_data_domain, alpha=alpha, q='mode', noise_cov=gal_stddev**2).get_model()
        # build the full model and connect it to the likelihood

            implicit_model = implicit_response @ galactic_model.get_model()
            residual = ift.Adder(-gal_rm) @ implicit_model
            new_dom = ift.MultiDomain.make({'icov': implicit_noise.target, 'residual': residual.target})
            n_res = ift.FieldAdapter(new_dom, 'icov')(implicit_noise.reciprocal()) + \
                ift.FieldAdapter(new_dom, 'residual')(residual)
            implicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=gal_data_domain, residual_key='residual',
                                                                inverse_covariance_key='icov',
                                                                   sampling_dtype=np.dtype(np.float64)) @ n_res

        else:
        #to use with perfect noise knowledge
            implicit_noise = Egf.StaticNoise(gal_data_domain, gal_stddev**2, True)

        # build the full model and connect it to the likelihood

            implicit_model = implicit_response @ galactic_model.get_model()
            residual = ift.Adder(-gal_rm) @ implicit_model
            implicit_likelihood = ift.GaussianEnergy(inverse_covariance=implicit_noise.get_model(),
                                                sampling_dtype=float) @ residual

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
            'plot_path': params['params_inference.plot_path'],
            'likelihoods': likelihoods,
            'sky_maps': sky_models,
            'power_spectra': power_models,
            'scatter_pairs': None,
            'plotting_kwargs': plotting_kwargs
        }

        return {'minimizer_params': minimizer_params, 'ecomponents': ecomponents}