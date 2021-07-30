import nifty7 as ift


def faraday_2020(param_dict):
    # Define Domain

    nside = param_dict['nside']
    hp = ift.makeDomain(ift.HPSpace(nside))

    # Define log-amplitude field
    rho_hyper_parameters = param_dict['rho_hyper_parameters']

    rho = ift.CorrelatedFieldMaker(prefix='rho')
    rho.add_fluctuations(**rho_hyper_parameters['fluctuations'])
    rho.set_amplitude_total_offset(**rho_hyper_parameters['offset'])

    rho_model = rho.finalize()

    # Define sign field
    chi_hyper_parameters = param_dict['chi_hyper_parameters']

    chi = ift.CorrelatedFieldMaker(prefix='chi')
    chi.add_fluctuations(**chi_hyper_parameters['fluctuations'])
    chi.set_amplitude_total_offset(**chi_hyper_parameters['offset'])

    chi_model = chi.finalize()

    # Build Faraday sky
    galactic_faraday_sky = rho_model.exp()*chi_model

    return {'sky':  galactic_faraday_sky,
            'components': {'rho': rho_model, 'chi': chi_model},
            'amplitudes': {'rho': rho.amplitude, 'chi': chi.amplitude}
            }
