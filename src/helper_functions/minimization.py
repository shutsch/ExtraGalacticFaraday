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
    ident = str(i)
    
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

    sample_parameters = {'n': Egf.config['sample_params']['n'],
                         'change_params': {'n_prior': Egf.config['sample_params']['n_prior'],
                                           'n_final': Egf.config['sample_params']['n_final'],
                                           'increase_step': Egf.config['sample_params']['increase_step'],
                                           'increase_rate': Egf.config['sample_params']['increase_rate']
                                           }
                         }

    controller_parameters = {
        'Sampler':
            {'n': Egf.config['controllers']['sampler']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': Egf.config['controllers']['sampler']['n_final'],
                               'increase_step': Egf.config['controllers']['sampler']['increase_step'],
                               'increase_rate': Egf.config['controllers']['sampler']['increase_rate']
                               },
             'controller_params': {'deltaE': Egf.config['controllers']['sampler']['deltaE'],
                                   'convergence_level': Egf.config['controllers']['sampler']['convergence_level']}
             },
        'Minimizer':
            {'n': Egf.config['controllers']['minimizer']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': Egf.config['controllers']['minimizer']['n_final'],
                               'increase_step': Egf.config['controllers']['minimizer']['increase_step'],
                               'increase_rate': Egf.config['controllers']['minimizer']['increase_rate']
                               },
             'controller_params': {'deltaE': Egf.config['controllers']['minimizer']['deltaE'],
                                   'convergence_level': Egf.config['controllers']['minimizer']['convergence_level']} #maybe at 2
             },
        'Minimizer_Samples':
            {'n': Egf.config['controllers']['minimizer_samples']['n'],
             'type': 'AbsDeltaEnergy',
             'change_params': {'n_final': Egf.config['controllers']['minimizer_samples']['n_final'],
                               'increase_step': Egf.config['controllers']['minimizer_samples']['increase_step'],
                               'increase_rate': Egf.config['controllers']['minimizer_samples']['increase_rate']
                               },
             'controller_params': {'deltaE': Egf.config['controllers']['minimizer_samples']['deltaE'],
                                   'convergence_level': Egf.config['controllers']['minimizer_samples']['convergence_level']}
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
    #kwargs={'mean':0., 'std': 1.0}
    #p_d['chi_int_0'] = ift.full(chiint0_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(),**kwargs).val.item()) 
    #p_d['chi_env_0'] = ift.full(chienv0_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(), **kwargs).val.item()) 
    #p_d['chi_lum'] = ift.full(chilum_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(), **kwargs).val.item()) 
    #p_d['chi_red'] = ift.full(chired_field.domain, ift.from_random(domain= ift.DomainTuple.scalar_domain(), **kwargs).val.item()) 
    
    
    #position = position.from_dict(p_d)

    #p_d = position.to_dict() 
    #chiint0_field = p_d['chi_int_0'] 
    #chienv0_field = p_d['chi_env_0'] 
    #p_d['chi_int_0'] = ift.from_random(domain= chiint0_field.domain,std=2).val.item() 
    #p_d['chi_env_0'] = ift.from_random(domain= chienv0_field.domain,std=2).val.item() 
    #position = position.from_dict(p_d)

    # n_samples = lambda iiter: 10 if iiter < 5 else 20

    n_samples = lambda i: get_n_samples(sample_parameters, 0, False) if i<99 else 100

  
    constants = lambda i: ['chi_lum', 'chi_int_0', 'chi_red', 'chi_env_0'] if i<10 \
        else (['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'noise_excitations', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'] if 10<=i<=80 \
            else [])

    point_estimates = lambda i: ['chi_lum', 'chi_int_0', 'chi_red', 'chi_env_0']  if i<10 \
        else (['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'noise_excitations', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'] if 10<=i<=80 \
            else [])
    
    op_output = {}
    sample_list, mean = ift.optimize_kl(
        likelihood_energy=likelihood,
        total_iterations=n_global,
        constants=constants,
        point_estimates=point_estimates,
        #n_samples=get_n_samples(sample_parameters, 0, False),
        n_samples = n_samples,
        kl_minimizer=ift.NewtonCG(controllers['Minimizer']),
        sampling_iteration_controller=controllers['Sampler'],
        nonlinear_sampling_minimizer=None,
        export_operator_outputs=op_output,
        initial_position=position,
        return_final_position=True,
        inspect_callback=plot_cb,
        output_directory='./runs/demo/results/'
        #dry_run=False
        )

