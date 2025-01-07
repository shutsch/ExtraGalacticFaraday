import nifty8 as ift

from .logger import logger, Format

from .minimization_helpers import get_controller, get_n_samples
from .plot.plot import sky_map_plotting, power_plotting, energy_plotting, scatter_plotting_posterior
import libs as Egf

_localParams = {}
_params = {}
_controllerParameters = {}
_controllers = {}


class Minimizer():
    def __init__(self, minimizer_params, params):
        self.minimizer_params = minimizer_params
        self.params = params

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
    def minimize(self):

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
                    scatter_plotting_posterior(sc1, sc2, key, _localParams['plot_path'], [s for s in latest_sample_list.iterator()], string=ident,
                                        **_localParams['plotting_kwargs'].get(key, {}))

            # minimizer = ift.NewtonCG(controllers['Minimizer'])
            # kl, _ = minimizer(kl)
            # final = True if i == _localParams['n_global'] - 1 else False
            # controllers = {key: get_controller(controller_dict, i, final, key)
            #                 for key, controller_dict in _controllerParameters.items()}

            # position = latest_mean

        def get_minimizer(i):
            if i<100: 
                deltaE_threshold = _params['controllers.minimizer.deltaE_threshold']

                new_dict = {
                    'n': _params['controllers.minimizer.n'],
                        'type': 'AbsDeltaEnergy',
                        'change_params': {'n_final': _params['controllers.minimizer.n_final'],
                                        'increase_step': _params['controllers.minimizer.increase_step'],
                                        'increase_rate': _params['controllers.minimizer.increase_rate']
                                        },
                        'controller_params': {'deltaE': _params['controllers.minimizer.deltaE_start'] 
                                            if i <= deltaE_threshold 
                                            else _params['controllers.minimizer.deltaE_end'],
                                            'convergence_level': _params['controllers.minimizer.convergence_level']}
                        }

                new_controller = get_controller(new_dict, i, False, 'Minimizer')
                return ift.NewtonCG(new_controller)  
            else:
                return ift.NewtonCG(_controllers['Minimizer_eg'])

        
        global _localParams, _params
        _localParams = self.minimizer_params
        # _localParams = {
        #     'sky_maps':self.minimizer_params['sky_maps'],
        #     'n_global':self.minimizer_params['n_global'],
        #     'power_spectra':self.minimizer_params['power_spectra'],
        #     'scatter_pairs':self.minimizer_params['scatter_pairs'],
        #     'plotting_kwargs':self.minimizer_params['plotting_kwargs'],
        #     'plot_path': self.minimizer_params['plot_path'],
        #     'likelihoods': self.minimizer_params['likelihoods']

        # }
        _params = self.params

        plotting_kwargs = self.minimizer_params['plotting_kwargs']
        likelihoods = self.minimizer_params['likelihoods']
        n_global = self.minimizer_params['n_global']

        sample_parameters = {'n': self.params['sample_params.n'],
                            'change_params': {'n_prior': self.params['sample_params.n_prior'],
                                            'n_final': self.params['sample_params.n_final'],
                                            'increase_step': self.params['sample_params.increase_step'],
                                            'increase_rate': self.params['sample_params.increase_rate']
                                            }
                            }

        controller_parameters = {
            'Sampler':
                {'n': self.params['controllers.sampler.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': self.params['controllers.sampler.n_final'],
                                'increase_step': self.params['controllers.sampler.increase_step'],
                                'increase_rate': self.params['controllers.sampler.increase_rate']
                                },
                'controller_params': {'deltaE': self.params['controllers.sampler.deltaE'],
                                    'convergence_level': self.params['controllers.sampler.convergence_level']}
                },
            'Minimizer':
                {'n': self.params['controllers.minimizer.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': self.params['controllers.minimizer.n_final'],
                                'increase_step': self.params['controllers.minimizer.increase_step'],
                                'increase_rate': self.params['controllers.minimizer.increase_rate']
                                },
                'controller_params': {'deltaE': self.params['controllers.minimizer.deltaE_start'],
                                    'convergence_level': self.params['controllers.minimizer.convergence_level']} #maybe at 2
                },
            'Sampler_eg':
                {'n': self.params['controllers.sampler_eg.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': self.params['controllers.sampler_eg.n_final'],
                                'increase_step': self.params['controllers.sampler_eg.increase_step'],
                                'increase_rate': self.params['controllers.sampler_eg.increase_rate']
                                },
                'controller_params': {'deltaE': self.params['controllers.sampler_eg.deltaE'],
                                    'convergence_level': self.params['controllers.sampler_eg.convergence_level']}
                },
            'Minimizer_eg':
                {'n': self.params['controllers.minimizer_eg.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': self.params['controllers.minimizer_eg.n_final'],
                                'increase_step': self.params['controllers.minimizer_eg.increase_step'],
                                'increase_rate': self.params['controllers.minimizer_eg.increase_rate']
                                },
                'controller_params': {'deltaE': self.params['controllers.minimizer_eg.deltaE'],
                                    'convergence_level': self.params['controllers.minimizer_eg.convergence_level']} #maybe at 2
                },
            'Minimizer_Samples':
                {'n': self.params['controllers.minimizer_samples.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': self.params['controllers.minimizer_samples.n_final'],
                                'increase_step': self.params['controllers.minimizer_samples.increase_step'],
                                'increase_rate': self.params['controllers.minimizer_samples.increase_rate']
                                },
                'controller_params': {'deltaE': self.params['controllers.minimizer_samples.deltaE'],
                                    'convergence_level': self.params['controllers.minimizer_samples.convergence_level']}
                },

        }

        global _controllerParameters
        _controllerParameters = controller_parameters

        if  plotting_kwargs is None:
            plotting_kwargs = {}

        controllers = {key: get_controller(controller_dict, 0, False, key)
                    for key, controller_dict in controller_parameters.items()}
        global _controllers
        _controllers = controllers

        # build the  likelihood
        likelihood = ift.utilities.my_sum(likelihoods.values())

        # draw position

        position = 0.1*ift.from_random(likelihood.domain)

        n_samples= lambda i: get_n_samples(sample_parameters, 0, False) if i< n_global-1 \
            else self.params['params.n_samples_posterior']


        constants = lambda i: [] if i< self.params['params.n_single_fit'] \
            else ( ['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'])

        point_estimates = lambda i: [] if i< self.params['params.n_single_fit'] \
            else (['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'])
        
        get_sampler = lambda i : controllers['Sampler'] if i< self.params['params.n_single_fit'] else controllers['Sampler_eg']
    
        op_output = {}
        sample_list, mean = ift.optimize_kl(
            likelihood_energy=likelihood,
            total_iterations=n_global,
            constants=constants,
            point_estimates=point_estimates,
            n_samples = n_samples,
            kl_minimizer=get_minimizer,
            sampling_iteration_controller=get_sampler,
            nonlinear_sampling_minimizer=None,
            export_operator_outputs=op_output,
            initial_position=position,
            return_final_position=True,
            inspect_callback=plot_cb,
            output_directory=self.params['params.results_path'],
            resume=True
            #dry_run=False
            )

