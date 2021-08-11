import nifty7 as ift
import ExtraGalacticFaraday as EgF

def run_inference():

    # set the HealPix resolution parameter and the sky domain

    nside = 256
    sky_domain = ift.makeDomain(ift.HPSpace(nside))

    # set the sky model hyper-parameters and initialize the Faraday 2020 sky model

    log_amplitude_params = {}

    sign_params = {}

    galactic_model = EgF.build_faraday_2020(sky_domain , **{'log_amplitude' : log_amplitude_params, 'sign': sign_params})

    # load_the data, define domains, covariance and projection operators

    data = EgF.get_rm(filter_pulsars=True, version='0.1.8', default_error_level=0.5).values()

    # filter
    schnitzeler_indices =  data['catalog'] == '2017MNRAS.467.1776K'

    #
    egal_rm = data['rm'][schnitzeler_indices]
    egal_stddev = data['rm_err'][schnitzeler_indices]

    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)

    explicit_response = EgF.SkyProjector(theta=data['theta'][egal_indices], phi=data['phi'][egal_indices], domain=sky_domain,
                                     target=egal_data_domain)

    egal_inverse_noise = EgF.StaticNoise(egal_data_domain, egal_stddev**2, True)

    # set the extra-galactic model hyper-parameters and initialize the model

    model_params = {'mu_a': 1,
                    'sigma_a': 1,
                    'mu_b': 1,
                    'sigma_b': 1,
                    }

    emodel = EgF.ExtraGalDemoModel(egal_data_domain, **model_params)

    # build the full model and connect it to the likelihood

    egal_model = egal_response @ galactic_model.get_model() + emodel.get_model()

    explicit_likelihood = ift.GaussianEnergy(mean=egal_model, inverse_covariance=egal_inverse_noise) @ egal_model


    egal_rm = data['rm'][egal_indices]
    egal_stddev = data['rm_err'][egal_indices]

    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)

    egal_response = EgF.SkyProjector(theta=data['theta'][egal_indices], phi=data['phi'][egal_indices], domain=sky_domain,
                                     target=egal_data_domain)

    implicit_inverse_noise = EgF.StaticNoise(egal_data_domain, egal_stddev**2, True)

    # build the full model and connect it to the likelihood

    implicit_model = implicit_response @ galactic_model.get_model()

    implicit_likelihood = ift.GaussianEnergy(mean=egal_model, inverse_covariance=implicit_inverse_noise) @ egal_model

    # combine the likelihoods

    likelihood = implicit_likelihood + explicit_likelihood

    # set run parameters and start the inference

    plotting_path =
    kl_type = 'GeoMetricKL'
    sampling_controller = ift.AbsDeltaEnergyController()
    minimization_controller = ift.AbsDeltaEnergyController()




if __name__ == '__main__ ':
    run_inference()