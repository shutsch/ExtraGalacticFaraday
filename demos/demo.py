import nifty7 as ift
import ExtraGalacticFaraday as egf

def run_inference():

    # set the HealPix resolution parameter and the sky domain

    nside = 256
    sky_domain = ift.makeDomain(ift.HPSpace(nside))

    # load_the data, define domains, covariance and projection operators

    rm_data, rm_stddev, theta, phi = egf.get_rm(filter_pulsars=True, filter_cgps=True, version='0.1.8')

    data_domain = ift.makeDomain(ift.UnstructuredDomain((len(rm_data),)))

    rm_data = ift.Field(data_domain, rm_data)
    rm_stddev = ift.Field(data_domain, rm_stddev)

    response = egf.SkyProjector(theta=theta, phi=phi, domain=sky_domain, target=data_domain)
    inverse_noise_cov = ift.makeOp(rm_stddev**(-2))

    # set the sky model hyper-parameters and initialize the model

    log_amplitude_params = {}

    sign_params = {}

    egf.faraday_2020({'log_amplitude' : log_amplitude_params, })

    # set the extra-galactic model hyper-parameters and initialize the model

    # build the full model and connect it to the likelihood


    full_model = response_op @ galactic_model + extra_galactic_model

    likelihood = ift.GaussianEnergy





if __name__ == '__main__ ':
    run_inference()