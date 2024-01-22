import nifty8 as ift
from .minimization_helpers import get_controller, get_n_samples
from .plot.plot import sky_map_plotting, power_plotting, energy_plotting, scatter_plotting
import libs as Egf


def minimization(likelihoods, kl_type, n_global, plot_path, sky_maps=None, power_spectra=None, scatter_pairs=None,
                 plotting_kwargs=None):
    """
    :param plot_path: str
        path where plots and results are stored.
    :param likelihoods: list of ift.Operators
        the likelihoods
    :param kl_type: str
        either 'MGVI' or
    :param n_global: int
        number of global iterations
    :param sky_maps: dict of ift.Operators
        sky map models for plotting
    :param power_spectra: dict of ift.Operators
        power spectrum models for plotting
    :param scatter_pairs:
    :return: None
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
            {'n': Egf.config['controllers']['sampler']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 100,
                               'increase_step': None,
                               'increase_rate': None
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },
        'Minimizer':
            {'n': Egf.config['controllers']['minimizer']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 40,
                               'increase_step': None,
                               'increase_rate': None
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },
        'Minimizer_Samples':
            {'n': Egf.config['controllers']['minimizer_samples']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 20,
                               'increase_step': True,
                               'increase_rate': True
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },

    }

    if plotting_kwargs is None:
        plotting_kwargs = {}

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
        if kl_type == 'SampledKLEnergy':
            kl_dict.update({'minimizer_sampling': ift.NewtonCG(controllers['Minimizer_Samples'])})
        #VERY HEAVY OPERATION
        kl = getattr(ift, kl_type)(position=position, hamiltonian=hamiltonian, **kl_dict)
        energy_dict.update({key: energy_dict[key] + [likelihoods[key].force(kl.position).val, ] for key in likelihoods})
        energy_plotting(energy_dict, plot_path)
        ident = str(i - 1) if i != 0 else 'initial'

        if sky_maps is not None:
            for sky_name, sky in sky_maps.items():
                sky_map_plotting(sky, [kl.position + s for s in kl.samples], sky_name, plot_path, string=ident,
                                 **plotting_kwargs.get(sky_name, {}))
                if sky_name not in power_spectra:
                    power_plotting(sky, [kl.position + s for s in kl.samples], sky_name, plot_path, string=ident,
                                   from_power_model=False, **plotting_kwargs.get(sky_name, {}))
        if power_spectra is not None:
            for power_name, power in power_spectra.items():
                power_plotting(power, [kl.position + s for s in kl.samples], power_name, plot_path, string=ident,
                               from_power_model=True,  **plotting_kwargs.get(power_name, {}))

        if scatter_pairs is not None:
            for key, (sc1, sc2) in scatter_pairs.items():
                scatter_plotting(sc1, sc2, key, plot_path, [kl.position + s for s in kl.samples], string=ident,
                                 **plotting_kwargs.get(key, {}))

        minimizer = ift.NewtonCG(controllers['Minimizer'])
        kl, _ = minimizer(kl)
        final = True if i == n_global - 1 else False
        controllers = {key: get_controller(controller_dict, i, final, key)
                       for key, controller_dict in controller_parameters.items()}

        position = kl.position
