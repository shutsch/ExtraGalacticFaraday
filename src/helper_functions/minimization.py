import nifty8 as ift
from .minimization_helpers import get_controller, get_n_samples
from .plot.plot import sky_map_plotting, power_plotting, energy_plotting, scatter_plotting
import libs as Egf

_localParams = []
_controllerParameters = {}

def plot_cb(latest_sample_list, i):
    latest_mean = latest_sample_list.average()
    
    energy_dict = {key: list() for key in _localParams['likelihoods']}
    energy_dict.update({key: energy_dict[key] + [_localParams['likelihoods'][key].force(latest_mean).val, ] for key in _localParams['likelihoods']})
    
    energy_plotting(energy_dict, _localParams['plot_path'])
    ident = str(i - 1) if i != 0 else 'initial'
    
    if _localParams['sky_maps'] is not None:
        for sky_name, sky in _localParams['sky_maps'].items():
            sky_map_plotting(sky, [s for s in latest_sample_list.iterator()], sky_name, _localParams['plot_path'], string=ident,
                                **_localParams['plotting_kwargs'].get(sky_name, {}))
            if sky_name not in _localParams['power_spectra']:
                power_plotting(sky, [s for s in latest_sample_list.iterator()], sky_name, _localParams['plot_path'], string=ident,
                                from_power_model=False, **_localParams['plotting_kwargs'].get(sky_name, {}))
    if _localParams['power_spectra'] is not None:
        for power_name, power in _localParams['power_spectra'].items():
            power_plotting(power, [s for s in latest_sample_list.iterator()], power_name, _localParams['plot_path'], string=ident,
                            from_power_model=True,  **_localParams['plotting_kwargs'].get(power_name, {}))

    if _localParams['scatter_pairs'] is not None:
        for key, (sc1, sc2) in _localParams['scatter_pairs'].items():
            scatter_plotting(sc1, sc2, key, _localParams['plot_path'], [s for s in latest_sample_list.iterator()], string=ident,
                                **_localParams['plotting_kwargs'].get(key, {}))

    # minimizer = ift.NewtonCG(controllers['Minimizer'])
    # kl, _ = minimizer(kl)
    # final = True if i == _localParams['n_global'] - 1 else False
    # controllers = {key: get_controller(controller_dict, i, final, key)
    #                 for key, controller_dict in _controllerParameters.items()}

    # position = latest_mean


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

    global _localParams
    _localParams = {
        'sky_maps':sky_maps,
        'n_global':n_global,
        'power_spectra':power_spectra,
        'scatter_pairs':scatter_pairs,
        'plotting_kwargs':plotting_kwargs,
        'plot_path': plot_path,
        'likelihoods': likelihoods

    }

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
             'change_params': {'n_final': 500,
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
                                   'convergence_level': 1} #maybe at 2
             },
        'Minimizer_Samples':
            {'n': Egf.config['controllers']['minimizer_samples']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': 80,
                               'increase_step': None,
                               'increase_rate': None
                               },
             'controller_params': {'deltaE': 1.0e-07,
                                   'convergence_level': 1}
             },

    }

    global _controllerParameters
    _controllerParameters = controller_parameters

    if plotting_kwargs is None:
        plotting_kwargs = {}

    controllers = {key: get_controller(controller_dict, 0, False, key)
                   for key, controller_dict in controller_parameters.items()}

    # build the  likelihood
    likelihood = ift.utilities.my_sum(likelihoods.values())

    # draw position

    position = 0.1*ift.from_random(likelihood.domain)

    #p_d = position.to_dict() 
    #chiint0_field = p_d['chi_int_0'] 
    #chienv0_field = p_d['chi_env_0'] 
    #chilum_field = p_d['chi_lum'] 
    #chired_field = p_d['chi_red'] 
    #p_d['chi_int_0'] = ift.full(chiint0_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(),random_type='uniform', low=-10,high=10).val.item()) 
    #p_d['chi_env_0'] = ift.full(chienv0_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(),random_type='uniform', low=-10,high=10).val.item()) 
    #p_d['chi_lum'] = ift.full(chilum_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(),random_type='normal', mean=0., std= 1.0).val.item()) 
    #p_d['chi_red'] = ift.full(chired_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(),random_type='normal', mean=0., std= 1.0).val.item()) 
    #position = position.from_dict(p_d)

    #p_d = position.to_dict() 
    #chiint0_field = p_d['chi_int_0'] 
    #chienv0_field = p_d['chi_env_0'] 
    #p_d['chi_int_0'] = ift.from_random(domain= chiint0_field.domain,std=2).val.item() 
    #p_d['chi_env_0'] = ift.from_random(domain= chienv0_field.domain,std=2).val.item() 
    #position = position.from_dict(p_d)


    op_output = {}
    sample_list, mean = ift.optimize_kl(
        likelihood_energy=likelihood,
        total_iterations=n_global,
        n_samples=get_n_samples(sample_parameters, 0, False),
        kl_minimizer=ift.NewtonCG(controllers['Minimizer']),
        sampling_iteration_controller=controllers['Sampler'],
        nonlinear_sampling_minimizer=None,
        export_operator_outputs=op_output,
        initial_position=position,
        return_final_position=True,
        inspect_callback=plot_cb
        output_directory='runs/demo/'
        #dry_run=False
        )

