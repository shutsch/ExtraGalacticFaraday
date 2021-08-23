import nifty7 as ift
import ExtraGalacticFaraday as EgF

def run_inference():

    # set the HealPix resolution parameter and the sky domain

    nside = 32
    sky_domain = ift.makeDomain(ift.HPSpace(nside))

    # set the sky model hyper-parameters and initialize the Faraday 2020 sky model

    log_amplitude_params = {}

    sign_params = {}

    galactic_model = EgF.Faraday2020Sky(sky_domain , **{'log_amplitude' : log_amplitude_params, 'sign': sign_params})

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
    explicit_likelihood = ift.GaussianEnergy(inverse_covariance=egal_inverse_noise.get_model()) \
                          @ residual


    gal_rm = data['rm'][~schnitzeler_indices]
    gal_stddev = data['rm_err'][~schnitzeler_indices]

    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(gal_rm),)))

    gal_rm = ift.Field(egal_data_domain, gal_rm)
    gal_stddev = ift.Field(egal_data_domain, gal_stddev)

    implicit_response = EgF.SkyProjector(theta=data['theta'][~schnitzeler_indices],
                                         phi=data['phi'][~schnitzeler_indices],
                                         domain=sky_domain, target=gal_data_domain)

    implicit_noise = EgF.SimpleVariableNoise(gal_data_domain, alpha=2.5, q='mode', noise_cov=gal_stddev**2)

    # build the full model and connect it to the likelihood

    implicit_model = implicit_response @ galactic_model.get_model()
    residual = implicit_noise.get_model().sqrt().reciprocal() @ ift.Adder(-gal_rm) @ implicit_model
    implicit_likelihood = ift.GaussianEnergy(gal_data_domain) @ residual

    # combine the likelihoods

    likelihood = implicit_likelihood + explicit_likelihood

    # set run parameters and start the inference

    EgF.minimization(n_global=20, kl_type='GeoMetricKL', likelihood=likelihood)


if __name__ == '__main__ ':
    run_inference()