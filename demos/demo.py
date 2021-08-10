import nifty7 as ift
import ExtraGalacticFaraday as EgF

def run_inference():

    # set the HealPix resolution parameter and the sky domain

    nside = 256
    sky_domain = ift.makeDomain(ift.HPSpace(nside))

    # load_the data, define domains, covariance and projection operators

    rm_data, rm_stddev, theta, phi = EgF.get_rm(filter_pulsars=True, filter_cgps=True, version='0.1.8')

    data_domain = ift.makeDomain(ift.UnstructuredDomain((len(rm_data),)))

    rm_data = ift.Field(data_domain, rm_data)
    rm_stddev = ift.Field(data_domain, rm_stddev)

    response = EgF.SkyProjector(theta=theta, phi=phi, domain=sky_domain, target=data_domain)
    inverse_noise_cov = ift.makeOp(rm_stddev**(-2))

    # set the sky model hyper-parameters and initialize the model

    log_amplitude_params = {}

    sign_params = {}

    models_dict = EgF.build_faraday_2020(data_domain , **{'log_amplitude' : log_amplitude_params, 'sign': sign_params})

    galactic_model = models_dict['sky']

    # set the extra-galactic model hyper-parameters and initialize the model

    model_params = {'mu_a': 1,
                    'sigma_a': 1,
                    'mu_b': 1,
                    'sigma_b': 1,
                    }

    models_dict = EgF.build_demo_extra_gal(data_domain, **model_params)
    extra_galactic_model = models_dict['sky']

    # build the full model and connect it to the likelihood

    full_model = response @ galactic_model + extra_galactic_model

    likelihood = ift.GaussianEnergy(mean=rm_data, inverse_covariance=inverse_noise_cov) @ full_model

    # set run parameters and start the inference

    plotting_path =
    kl_type = 'GeoMetricKL'
    sampling_controller = ift.AbsDeltaEnergyController()
    minimization_controller = ift.AbsDeltaEnergyController()




if __name__ == '__main__ ':
    run_inference()