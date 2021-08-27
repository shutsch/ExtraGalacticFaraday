import nifty7 as ift
from .minimization_helpers import get_controller, get_n_samples
from .plot.plot import sky_map_plotting, power_plotting, energy_plotting


def minimization(likelihoods, kl_type, n_global, plot_path, sky_maps=None, power_spectra=None):
    """
    :param plot_path:
    :param likelihoods:
    :param kl_type:
    :param n_global:
    :param sky_maps:
    :param power_spectra:
    :return:
    """
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
    likelihood = ift.utilities.my_sum(likelihoods.values())

    hamiltonian = ift.StandardHamiltonian(likelihood, controllers['Sampler'])

    # build the minimization functional

    kl_dict = {'n_samples': get_n_samples(sample_parameters, 0, False), 'mirror_samples': True}

    position = ift.from_random(hamiltonian.domain)

    energy_dict = {key: list() for key in likelihoods}

    for i in range(n_global):
        if kl_type == 'GeoMetricKL':
            kl_dict.update({'minimizer_samp': ift.NewtonCG(controllers['Minimizer_Samples'])})
        kl = getattr(ift, kl_type)(mean=position, hamiltonian=hamiltonian, **kl_dict)
        energy_dict.update({key: energy_dict[key] + [likelihoods[key].force(kl.position).val, ] for key in likelihoods})
        energy_plotting(energy_dict, plot_path)

        if sky_maps is not None:
            for sky_name, sky in sky_maps.items():
                sky_map_plotting(sky, kl.samples, sky_name, str(i - 1) if i != 0 else 'initial', plot_path)
                if sky not in power_spectra:
                    power_plotting(sky,  sky_name, str(i - 1) if i != 0 else 'initial', plot_path,
                                   from_power_model=False)
        if power_spectra is not None:
            for power_name, power in power_spectra.items():
                power_plotting(power,  power_name, str(i - 1) if i != 0 else 'initial', plot_path,
                               from_power_model=True)

        minimizer = ift.NewtonCG(controllers['Minimizer'])
        kl, _ = minimizer(kl)
        final = True if i == n_global - 1 else False
        controllers = {key: get_controller(controller_dict, i, final, key)
                       for key, controller_dict in controller_parameters.items()}

        position = kl.position
