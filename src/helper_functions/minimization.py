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
            {'n': 100,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 100,
                               'increase_step': None,
                               'increase_rate': None
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },
        'Minimizer':
            {'n': 25,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 40,
                               'increase_step': None,
                               'increase_rate': None
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },
        'Minimizer_Samples':
            {'n': 25,
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 20,
                               'increase_step': True,
                               'increase_rate': True
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },

    }

    controllers = {key: get_controller(controller_dict, 0, False, key)
                   for key, controller_dict in controller_parameters.items()}

    # build the Hamiltonian
    hamiltonian = ift.StandardHamiltonian(likelihood, controllers['Sampler'])

    # build the minimization functional

    kl_dict = {'n_samples': get_n_samples(sample_parameters, 0, False), 'mirror_samples': True}

    position = ift.from_random(hamiltonian.domain)

    for i in range(n_global):
        if kl_type == 'GeoMetricKL':
            kl_dict.update({'minimizer_samp': ift.NewtonCG(controllers['Minimizer_Samples'])})
        kl = getattr(ift, kl_type)(mean=position, hamiltonian=hamiltonian, **kl_dict)
        minimizer = ift.NewtonCG(controllers['Minimizer'])
        kl, _ = minimizer(kl)
        final = True if i == n_global - 1 else False
        controllers = {key: get_controller(controller_dict, i, final, key)
                       for key, controller_dict in controller_parameters.items()}

        position = kl.mean






            



