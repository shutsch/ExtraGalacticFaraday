import nifty8 as ift
import libs as Egf
import numpy as np

# - fixed library requirements
# - set colorbar range to -250;+250


def run_inference():
    


    # set the HealPix resolution parameter and the sky domain

    sky_domain = ift.makeDomain(ift.HPSpace(Egf.config['params']['nside']))

    # load_the data, define domains, covariance and projection operators

    data = Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

    # filter
    #schnitzeler_indices = (data['catalog'] == '2017MNRAS.467.1776K')
    z_indices = ~np.isnan(data['z_best'])

    #
    #egal_rm = data['rm'][schnitzeler_indices]
    #egal_stddev = data['rm_err'][schnitzeler_indices]

    egal_rm = np.array(data['rm'][z_indices])
    egal_stddev = np.array(data['rm_err'][z_indices])
    egal_z = np.array(data['z_best'][z_indices])
    egal_L = np.array(data['stokesI'][z_indices])

    # set the sky model hyper-parameters and initialize the Faraday 2020 sky model
    #new parameters given by Sebastian
    log_amplitude_params = {'fluctuations': {'asperity': None,'flexibility': [.1, .1], 
                          'fluctuations': [1.0, 0.5], 'loglogavgslope': [-11/3, 1.],},
                            'offset': {'offset_mean': 5., 'offset_std': [1., 0.001]},}

    sign_params = {'fluctuations': {'asperity': None, 'flexibility': [.1, .1],
                    'fluctuations': [5.0, 4.0], 'loglogavgslope': [-11/3, 1.0], },
                   'offset': {'offset_mean': 0, 'offset_std': [5., 4.]},}
    
   

    galactic_model = Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)
    

    

    # build the full model and connect it to the likelihood
    # set the extra-galactic model hyper-parameters and initialize the model
    egal_model_params = {'z': egal_z,'L': egal_L, 'alpha': 1.0, 'q': 0.01
         }
      
    emodel = Egf.ExtraGalDemoModel(egal_data_domain, egal_model_params)

    #if we are not interested in the RM but only in its sigma we can consider the eg sigma as a noise and sum the two here. 
    #we include it here but not in the Variable Noise below because the variable noise include the eta factors and applies only to
    #the Tayolor catalog. Here we are considering the LOFAR catalog. When we will include the correlated eg component, the line 
    #below will include again only the noise. 
    

    noise_params = {
        'egal_var': egal_stddev**2,
        'emodel': emodel.get_model()
    }

    egal_inverse_noise = Egf.EgalAddingNoise(egal_data_domain, noise_params).get_model()


    
    explicit_response = Egf.SkyProjector(theta=data['theta'][z_indices], phi=data['phi'][z_indices],
                                         domain=sky_domain, target=egal_data_domain) 

      
    #if we are not interested in the RM but only in its sigma we do not need to include the Rm in the following line
    egal_model = explicit_response @ galactic_model.get_model()
    #egal_model = explicit_response @ galactic_model.get_model() + emodel.get_model()
    residual = ift.Adder(-egal_rm) @ egal_model
    #we need to use the VariableCovarianceGaussianEnerg instead than the GaussianEnergy because the variance (that now
    #includes the eg part that now we are fitting) is varying, is not anymore a costant. When we will include the 
    #correlated eg component we will need to use again the GaussianEnergy. 
    new_dom = ift.MultiDomain.make({'icov': egal_inverse_noise.target, 'residual': residual.target})
    n_res = ift.FieldAdapter(new_dom, 'icov')(egal_inverse_noise.reciprocal()) + \
        ift.FieldAdapter(new_dom, 'residual')(residual)
    explicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=egal_data_domain, residual_key='residual',
                                                               inverse_covariance_key='icov',
                                                               sampling_dtype=np.dtype(np.float64)) @ n_res
    
    #explicit_likelihood = ift.VariableCovarianceGaussianEnergy(inverse_covariance=egal_inverse_noise.get_model()+emodel.get_model(),
    #                                         sampling_dtype=float) @ residual
    #explicit_likelihood = ift.GaussianEnergy(inverse_covariance=egal_inverse_noise.get_model(),
    #                                         sampling_dtype=float) @ residual


    gal_rm = np.array(data['rm'][~z_indices])
    gal_stddev = np.array(data['rm_err'][~z_indices])

  
    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(gal_rm),)))

    gal_rm = ift.Field(gal_data_domain, gal_rm)
    gal_stddev = ift.Field(gal_data_domain, gal_stddev)

    implicit_response = Egf.SkyProjector(theta=data['theta'][~z_indices],
                                         phi=data['phi'][~z_indices],
                                         domain=sky_domain, target=gal_data_domain)


    implicit_noise = Egf.SimpleVariableNoise(gal_data_domain, alpha=2.5, q='mode', noise_cov=gal_stddev**2).get_model()

    # build the full model and connect it to the likelihood

    implicit_model = implicit_response @ galactic_model.get_model()
    residual = ift.Adder(-gal_rm) @ implicit_model
    new_dom = ift.MultiDomain.make({'icov': implicit_noise.target, 'residual': residual.target})
    n_res = ift.FieldAdapter(new_dom, 'icov')(implicit_noise.reciprocal()) + \
        ift.FieldAdapter(new_dom, 'residual')(residual)
    implicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=gal_data_domain, residual_key='residual',
                                                               inverse_covariance_key='icov',
                                                               sampling_dtype=np.dtype(np.float64)) @ n_res

    # set run parameters and start the inference
    components = galactic_model.get_components()
    ecomponents = emodel.get_components()

    sky_models = {'faraday_sky': galactic_model.get_model(), 'profile': components['log_profile'].exp(),
                  'sign': components['sign']}
    power_models = {'log_profile': components['log_profile_amplitude'], 'sign': components['sign_amplitude']}
    #scatter_pairs = {'egal_results_vs_data': (egal_model, egal_rm)}
    #scatter_pairs = None
    #scatter_pairs = {'intrinsic': (ecomponents['chi_lum'], ecomponents['sigma_int_0']),'environmental': (ecomponents['chi_red'], ecomponents['sigma_env_0'])}
    
    #the value that we plot are indeed the values in the position field 
    scatter_pairs = {'intrinsic': (ecomponents['chi_lum'], ecomponents['sigma_int_02']),'environmental': (ecomponents['chi_red'], ecomponents['sigma_env_02'])}

    #plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
    #                                   'vmin_mean':'-250', 'vmax_mean':'250', 
    #                                   'vmin_std':'-250', 'vmax_std':'250'},
    #                   'egal_results_vs_data': {'x_label': 'results', 'y_label': 'data'}}
    plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
                                       'vmin_mean':'-250', 'vmax_mean':'250', 
                                       'vmin_std':'-250', 'vmax_std':'250'},
                       'intrinsic': {'x_label': 'chi_lum', 'y_label': 'sigma_int_0'},
                       'environmental': {'x_label': 'chi_red', 'y_label': 'sigma_env_0'}}

    Egf.minimization(n_global=Egf.config['params']['nglobal'], kl_type='SampledKLEnergy', plot_path=Egf.config['params']['plot_path'],
                     likelihoods={'implicit_likelihood': implicit_likelihood,
                                  'explicit_likelihood': explicit_likelihood},
                     sky_maps=sky_models, power_spectra=power_models, scatter_pairs=scatter_pairs,
                     plotting_kwargs=plotting_kwargs)


if __name__ == '__main__':
    # print a RuntimeWarning  in case of underflows
    np.seterr(under='warn') 
    # set seed
    seed = 1000
    ift.random.push_sseq_from_seed(seed)
    run_inference()
