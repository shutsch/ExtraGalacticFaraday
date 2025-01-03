import nifty8 as ift
import libs as Egf
import numpy as np
from plot_posterior import Posterior_Plotter
from src.helper_functions.parameters_maker import Parameters_maker
import utilities as U


def run_inference(params):
    


    # set the HealPix resolution parameter and the sky domain

    sky_domain = ift.makeDomain(ift.HPSpace(params['params.nside']))

    # load_the data, define domains, covariance and projection operators

    data = Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

    # filter
    z_indices = ~np.isnan(data['z_best'])

    e_rm = np.array(data['rm'][z_indices])
    e_stddev = np.array(data['rm_err'][z_indices])
    e_z = np.array(data['z_best'][z_indices])
    e_F = np.array(data['stokesI'][z_indices])

    # test parameters
    log_amplitude_params = {'fluctuations': {'asperity': params['params_mock_cat.log_amplitude.fluctuations.asperity'], 
                                            'flexibility': params['params_mock_cat.log_amplitude.fluctuations.flexibility'],  
                                            'fluctuations': params['params_mock_cat.log_amplitude.fluctuations.fluctuations'], 
                                            'loglogavgslope': params['params_mock_cat.log_amplitude.fluctuations.loglogavgslope'], },
                            'offset': {'offset_mean': params['params_mock_cat.log_amplitude.offset.offset_mean'], 
                                      'offset_std': params['params_mock_cat.log_amplitude.offset.offset_std']},}

    sign_params = {'fluctuations': {'asperity': params['params_mock_cat.sign.fluctuations.asperity'], 
                                            'flexibility': params['params_mock_cat.sign.fluctuations.flexibility'],  
                                            'fluctuations': params['params_mock_cat.sign.fluctuations.fluctuations'], 
                                            'loglogavgslope': params['params_mock_cat.sign.fluctuations.loglogavgslope'], },
                            'offset': {'offset_mean': params['params_mock_cat.sign.offset.offset_mean'], 
                                      'offset_std': params['params_mock_cat.sign.offset.offset_std']},}

    galactic_model = Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(e_rm),)))

    egal_rm = ift.Field(egal_data_domain, e_rm)
    egal_stddev = ift.Field(egal_data_domain, e_stddev)
    
    # build the full model and connect it to the likelihood
    # set the extra-galactic model hyper-parameters and initialize the model
    egal_model_params = {'z': e_z, 'F': e_F, 'n_params': params['params.n_eg_params'] }
      
    emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

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



    # Possible all sky variation of alpha, requires pygedm package 
    #alpha = 2.5
    #log_ymw = np.log(Egf.load_ymw_sky('./data/', nside=Egf.config['params']['nside'], model='ymw16', mode='mc'))
    #log_ymw /= log_ymw.min()
    #log_ymw *= 5
    #alpha = implicit_response(ift.Field(ift.makeDomain(implicit_response.domain), log_ymw)).val


    #implicit_noise = Egf.SimpleVariableNoise(gal_data_domain, alpha=alpha, q='mode', noise_cov=gal_stddev**2).get_model()
    implicit_noise = Egf.StaticNoise(gal_data_domain, gal_stddev**2, True)

    # build the full model and connect it to the likelihood

    implicit_model = implicit_response @ galactic_model.get_model()
    residual = ift.Adder(-gal_rm) @ implicit_model
    #new_dom = ift.MultiDomain.make({'icov': implicit_noise.target, 'residual': residual.target})
    #n_res = ift.FieldAdapter(new_dom, 'icov')(implicit_noise.reciprocal()) + \
    #    ift.FieldAdapter(new_dom, 'residual')(residual)
    #implicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=gal_data_domain, residual_key='residual',
    #                                                           inverse_covariance_key='icov',
    #                                                           sampling_dtype=np.dtype(np.float64)) @ n_res
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

    Egf.minimization(n_global=params['params.nglobal'], kl_type='SampledKLEnergy', plot_path=params['params.plot_path'],
                     likelihoods=likelihoods,
                     sky_maps=sky_models, power_spectra=power_models, #scatter_pairs=scatter_pairs,
                     plotting_kwargs=plotting_kwargs)
    
    plot_params = {'ecomponents': ecomponents, 'n_params': params['params.n_eg_params'], 'results_path':  params['params.results_path']}

    plotter = Posterior_Plotter(plot_params)


if __name__ == '__main__':
    params = Parameters_maker().get_parsed_params()

    # print a RuntimeWarning  in case of underflows
    np.seterr(all='raise')
    # set seed
    seed = params['params_mock_cat.maker_params.seed_inf']
    ift.random.push_sseq_from_seed(seed)
    run_inference(params)
