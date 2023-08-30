import nifty7 as ift
import libs as Egf
import numpy as np

# - fixed library requirements
# - set colorbar range to -250;+250


def run_inference():
    # set the HealPix resolution parameter and the sky domain

    sky_domain = ift.makeDomain(ift.HPSpace(Egf.config['params']['nside']))

    # load_the data, define domains, covariance and projection operators

    data = Egf.get_rm(filter_pulsars=True, version='0.1.8', default_error_level=0.5)

    # filter
    schnitzeler_indices = (data['catalog'] == '2017MNRAS.467.1776K')

    #
    egal_rm = data['rm'][schnitzeler_indices]
    egal_stddev = data['rm_err'][schnitzeler_indices]


    # set the sky model hyper-parameters and initialize the Faraday 2020 sky model

    log_amplitude_params = {'fluctuations': {'asperity': [.1, .1], 'flexibility': [.1, .1],
                                             'fluctuations': [3, 2], 'loglogavgslope': [-3., .75],
                                             },
                            'offset': {'offset_mean': 4, 'offset_std': [6, 6.]},
                            }

    sign_params = {'fluctuations': {'asperity': [.1, .1], 'flexibility': [.1, .1],
                                    'fluctuations': [3, 2], 'loglogavgslope': [-3., .75],
                                    },
                   'offset': {'offset_mean': 0, 'offset_std': [6, 6.]},
                   }

    galactic_model = Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)

    explicit_response = Egf.SkyProjector(theta=data['theta'][schnitzeler_indices], phi=data['phi'][schnitzeler_indices],
                                         domain=sky_domain, target=egal_data_domain)

    egal_inverse_noise = Egf.StaticNoise(egal_data_domain, egal_stddev**2, True)

    # set the extra-galactic model hyper-parameters and initialize the model

    
    L = data['l'][schnitzeler_indices]
    z = data['b'][schnitzeler_indices]
    egal_model_params = {
        'chi_lum': 1, 
        'chi_red': 1, 
        'sigma_int_0': 1, 
        'sigma_env_0': 1,
        'L': L,
        'z': z
        }

    emodel = Egf.ExtraGalDemoModel(egal_data_domain, egal_model_params)
    
    #TODO: new formula -> Rm^2 = (L/L0)^Xlum * sigma2_int_0/(1+z)^4 + D/D0 * sigma2_env_0
    ## D = integral 0 to z (c/H) * ((1+z)^(4 + Xred)) dz
    ## Xlum, Xred, sigma2_int_0, sigma2_env_0 to be provided in input, looping through values, in order
    ## to calculate different Rm^2, to be applied to Gaussian. Target is eg_contr (e_model)
    ## L0, D0 hyperpars (fixed), c hyperpar (speedlight), H hyperpar (depends on cosmology (refer to already existing code Valentina))
    # input: values from catalog (L, z)
    # output: rm^2, need to calculate eg_contr=G(0, rm^2) as output of function (see numpy.random.normal)
    # all outputs need to be put in some kind of array
    
    # build the full model and connect it to the likelihood

    egal_model = explicit_response @ galactic_model.get_model() + emodel.get_model()
    residual = ift.Adder(-egal_rm) @ egal_model
    explicit_likelihood = ift.GaussianEnergy(inverse_covariance=egal_inverse_noise.get_model(),
                                             sampling_dtype=float) @ residual

    gal_rm = data['rm'][~schnitzeler_indices]
    gal_stddev = data['rm_err'][~schnitzeler_indices]

    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(gal_rm),)))

    gal_rm = ift.Field(gal_data_domain, gal_rm)
    gal_stddev = ift.Field(gal_data_domain, gal_stddev)

    implicit_response = Egf.SkyProjector(theta=data['theta'][~schnitzeler_indices],
                                         phi=data['phi'][~schnitzeler_indices],
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
    sky_models = {'faraday_sky': galactic_model.get_model(), 'profile': components['log_profile'].exp(),
                  'sign': components['sign']}
    power_models = {'log_profile': components['log_profile_amplitude'], 'sign': components['sign_amplitude']}
    scatter_pairs = {'egal_results_vs_data': (egal_model, egal_rm)}
    # scatter_pairs = None

    plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
                                       'vmin_mean':'-250', 'vmax_mean':'250', 
                                       'vmin_std':'-250', 'vmax_std':'250'},
                       'egal_results_vs_data': {'x_label': 'results', 'y_label': 'data'}}

    Egf.minimization(n_global=Egf.config['params']['nglobal'], kl_type='GeoMetricKL', plot_path=Egf.config['params']['plot_path'],
                     likelihoods={'implicit_likelihood': implicit_likelihood,
                                  'explicit_likelihood': explicit_likelihood},
                     sky_maps=sky_models, power_spectra=power_models, scatter_pairs=scatter_pairs,
                     plotting_kwargs=plotting_kwargs)


if __name__ == '__main__':
    run_inference()
