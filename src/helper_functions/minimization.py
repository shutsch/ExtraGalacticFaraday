import nifty7 as ift
from .minimization_helpers import get_controller, get_n_samples


def minimization(likelihood, kl_type, n_global, **kwargs):
    sample_parameters = {'n': 2,
                         'change_params': {'n_prior': 2,
                                           'n_final': 20,
                                           'increase_step': 10,
                                           'increase_rate': 1.7
                                           }
                         }

    controller_parameters = {
        'Sampler':
            {'n': 2,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_prior': 2,
                               'n_final': 20,
                               'increase_step': None,
                               'increase_rate': None
                               }
             },
        'Minimizer':
            {'n': 2,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_prior': 2,
                               'n_final': 20,
                               'increase_step': None,
                               'increase_rate': None
                               }
             },
        'Minimizer_Samples':
            {'n': 2,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_prior': 2,
                               'n_final': 20,
                               'increase_step': True,
                               'increase_rate': True
                               }
             },

    }

    controllers = {key: get_controller(controller_dict, 0, False, key)
                   for key, controller_dict in controller_parameters.items()}

    # build the Hamiltonian
    hamiltonian = ift.StandardHamiltonian(likelihood, controllers['Sampler'])

    # build the minimization functional

    minimizer = ift.NewtonCG(controllers['Minimizer'])

    kl_dict = {'n_samples': get_n_samples(sample_parameters, 0, False), 'mirror_samples': True}
    if kl_type == 'GeoMetricKL':
        kl_dict.update({'minimizer_samp': controllers['Minimizer_Samples']})
    position = ift.from_random(hamiltonian.domain)

    for i in range(n_global):
        kl = getattr(ift, kl_type)(position, hamiltonian, **kl_dict)
        kl = minimizer(kl)
        final = True if i == n_global - 1 else False
        controllers = {key: get_controller(controller_dict, i, final, key)
                       for key, controller_dict in controller_parameters.items()}
        if kl_type == 'GeoMetricKL':
            kl_dict.update({'minimizer_samp': controllers['Minimizer_Samples']})
        position = kl.position






            



