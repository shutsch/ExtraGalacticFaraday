import nifty7 as ift

from ..Model import Model


class Faraday2020Sky(Model):
    def __init__(self, target_domain, log_amplitude_parameters, sign_parameters):
        super().__init__(target_domain)
        self._log_amplitude_parameters = log_amplitude_parameters
        self._sign_parameters = sign_parameters

    def set_model(self):

        rho = ift.CorrelatedFieldMaker(prefix='log_profile')
        rho.add_fluctuations(**self._log_amplitude_parameters['fluctuations'])
        rho.set_amplitude_total_offset(**self._log_amplitude_parameters['offset'])

        rho_model = rho.finalize()

        # Define sign field

        chi = ift.CorrelatedFieldMaker(prefix='sign')
        chi.add_fluctuations(**self._sign_parameters['fluctuations'])
        chi.set_amplitude_total_offset(**self._sign_parameters['offset'])

        chi_model = chi.finalize()

        # Build Faraday sky
        galactic_faraday_sky = rho_model.exp()*chi_model

        self._model = galactic_faraday_sky,
        self._components = {'log_profile': rho_model, 'sign': chi_model ,
                            'log_profile_amplitude': rho.amplitude, 'sign_amplitude': chi.amplitude
                            }
