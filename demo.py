import nifty8 as ift
import ExtraGalacticFaraday as EgF

import numpy as np

"""
"""


def run_inference():
    # set the HealPix resolution parameter and the sky domain

    nside = 32
    sky_domain = ift.makeDomain(ift.HPSpace(nside))

    # load_the data, define domains, covariance and projection operators

    data = EgF.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

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

    galactic_model = EgF.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)

    explicit_response = EgF.SkyProjector(theta=data['theta'][schnitzeler_indices], phi=data['phi'][schnitzeler_indices],
                                         domain=sky_domain, target=egal_data_domain)

    egal_inverse_noise = EgF.StaticNoise(egal_data_domain, egal_stddev**2, True)

    # set the extra-galactic model hyper-parameters and initialize the model

    egal_model_params = {'mu_a': 1, 'sigma_a': 1, 'mu_b': 1, 'sigma_b': 1,
                         }

    emodel = EgF.ExtraGalDemoModel(egal_data_domain, **egal_model_params)

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

    implicit_response = EgF.SkyProjector(theta=data['theta'][~schnitzeler_indices],
                                         phi=data['phi'][~schnitzeler_indices],
                                         domain=sky_domain, target=gal_data_domain)

    implicit_noise = EgF.SimpleVariableNoise(gal_data_domain, alpha=2.5, q='mode', noise_cov=gal_stddev**2).get_model()

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

    plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu'},
                       'egal_results_vs_data': {'x_label': 'results', 'y_label': 'data'}}

    EgF.minimization(n_global=20, kl_type='SampledKLEnergy', plot_path='./runs/demo/',
                     likelihoods={'implicit_lilelihood': implicit_likelihood,
                                  'explicit_likelihood': explicit_likelihood},
                     sky_maps=sky_models, power_spectra=power_models, scatter_pairs=scatter_pairs,
                     plotting_kwargs=plotting_kwargs)


if __name__ == '__main__':
    run_inference()
