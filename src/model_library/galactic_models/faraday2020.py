import nifty7 as ift


def build_faraday_2020(domain, log_amplitude_parameters, sign_parameters):

    # Define log-amplitude field

    rho = ift.CorrelatedFieldMaker(prefix='log_amplitude')
    rho.add_fluctuations(**log_amplitude_parameters['fluctuations'])
    rho.set_amplitude_total_offset(**log_amplitude_parameters['offset'])

    rho_model = rho.finalize()

    # Define sign field

    chi = ift.CorrelatedFieldMaker(prefix='sign')
    chi.add_fluctuations(**sign_parameters['fluctuations'])
    chi.set_amplitude_total_offset(**sign_parameters['offset'])

    chi_model = chi.finalize()

    # Build Faraday sky
    galactic_faraday_sky = rho_model.exp()*chi_model

    return {'sky':  galactic_faraday_sky,
            'components': {'log_amplitude': rho_model, 'sign': chi_model},
            'amplitudes': {'log_amplitude': rho.amplitude, 'sign': chi.amplitude}
            }
