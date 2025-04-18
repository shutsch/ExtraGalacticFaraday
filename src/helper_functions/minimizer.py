import nifty8 as ift
from .logger import logger, Format
from posterior_plotter import Posterior_Plotter
from .minimization_helpers import get_controller, get_n_samples
from .plot.plot import sky_map_plotting, power_plotting, energy_plotting, scatter_plotting_posterior
import numpy as np

class Minimizer():
    def __init__(self, minimizer_params, ecomponents, params):
        self.minimizer_params = minimizer_params
        self.ecomponents = ecomponents
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
            
            energy_dict = {key: list() for key in minimizer_params['likelihoods']}
            energy_dict.update({key: energy_dict[key] + [minimizer_params['likelihoods'][key].force(latest_mean).val, ] for key in minimizer_params['likelihoods']})
            
            energy_plotting(energy_dict, minimizer_params['plot_path'])
            ident = str(i)
            
            #TODO: samples = [s for s in latest_sample_list.iterator()] is the input
            #add a minimizer param with eta operator op, to be instanced with
            #op.force(samples).val
            #This can be appended to an empty list l, with l.append(op.force(samples).val.
            # Eventually, l_ar = np.array(l) produces a plottable data structure

            samples=[]
            latest_samples = [s for s in latest_sample_list.iterator()]
            eta_op = minimizer_params['eta']
            res = [eta_op.force(s) for s in latest_samples]
            samples.append(res)
            plo = ift.Plot()
            plo.add(samples, title="eta samples", linewidth=1, color='b')
            plo.output(name="eta samples.png")

            # if minimizer_params['eta'] is not None:
            #     eta_plotting(eta_op, samples)

            if minimizer_params['sky_maps'] is not None:
                for sky_name, sky in minimizer_params['sky_maps'].items():
                    sky_map_plotting(sky, [s for s in latest_sample_list.iterator()], sky_name, minimizer_params['plot_path'], string=ident,
                                        **minimizer_params['plotting_kwargs'].get(sky_name, {}))
                    if sky_name not in minimizer_params['power_spectra']:
                        power_plotting(sky, [s for s in latest_sample_list.iterator()], sky_name, minimizer_params['plot_path'], string=ident,
                                        from_power_model=False, **minimizer_params['plotting_kwargs'].get(sky_name, {}))
            
            
            if minimizer_params['power_spectra'] is not None:
                for power_name, power in minimizer_params['power_spectra'].items():
                    power_plotting(power, [s for s in latest_sample_list.iterator()], power_name, minimizer_params['plot_path'], string=ident,
                                    from_power_model=True,  **minimizer_params['plotting_kwargs'].get(power_name, {}))

            if minimizer_params['scatter_pairs'] is not None:
                for key, (sc1, sc2) in minimizer_params['scatter_pairs'].items():
                    scatter_plotting_posterior(sc1, sc2, key, minimizer_params['plot_path'], [s for s in latest_sample_list.iterator()], string=ident,
                                        **minimizer_params['plotting_kwargs'].get(key, {}))
            
            plot_params = {
                'ecomponents': self.ecomponents,
                'params': params,
            }
            
            Posterior_Plotter(plot_params).plot(figname=f'EG_posterior_{i}.png')

        def get_minimizer(i):
            if i<params['params_inference.n_single_fit']: 
                deltaE_threshold = params['controllers.minimizer.deltaE_threshold']

                new_dict = {
                    'n': params['controllers.minimizer.n'],
                        'type': 'AbsDeltaEnergy',
                        'change_params': {'n_final': params['controllers.minimizer.n_final'],
                                        'increase_step': params['controllers.minimizer.increase_step'],
                                        'increase_rate': params['controllers.minimizer.increase_rate']
                                        },
                        'controller_params': {'deltaE': params['controllers.minimizer.deltaE_start'] 
                                            if i <= deltaE_threshold 
                                            else params['controllers.minimizer.deltaE_end'],
                                            'convergence_level': params['controllers.minimizer.convergence_level']}
                        }

                new_controller = get_controller(new_dict, i, False, 'Minimizer')
                return ift.NewtonCG(new_controller)  
            else:
                return ift.NewtonCG(controllers['Minimizer_eg'])

        
        minimizer_params = self.minimizer_params
        params = self.params

        plotting_kwargs = minimizer_params['plotting_kwargs']
        likelihoods = minimizer_params['likelihoods']
        n_global = minimizer_params['n_global']

        sample_parameters = {'n': self.params['sample_params.n'],
                            'change_params': {'n_prior': params['sample_params.n_prior'],
                                            'n_final': params['sample_params.n_final'],
                                            'increase_step': params['sample_params.increase_step'],
                                            'increase_rate': params['sample_params.increase_rate']
                                            }
                            }

        controller_parameters = {
            'Sampler':
                {'n': params['controllers.sampler.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': params['controllers.sampler.n_final'],
                                'increase_step': params['controllers.sampler.increase_step'],
                                'increase_rate': params['controllers.sampler.increase_rate']
                                },
                'controller_params': {'deltaE': params['controllers.sampler.deltaE'],
                                    'convergence_level': params['controllers.sampler.convergence_level']}
                },
            'Minimizer':
                {'n': params['controllers.minimizer.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': params['controllers.minimizer.n_final'],
                                'increase_step': params['controllers.minimizer.increase_step'],
                                'increase_rate': params['controllers.minimizer.increase_rate']
                                },
                'controller_params': {'deltaE': params['controllers.minimizer.deltaE_start'],
                                    'convergence_level': params['controllers.minimizer.convergence_level']} #maybe at 2
                },
            'Sampler_eg':
                {'n': params['controllers.sampler_eg.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': params['controllers.sampler_eg.n_final'],
                                'increase_step': params['controllers.sampler_eg.increase_step'],
                                'increase_rate': params['controllers.sampler_eg.increase_rate']
                                },
                'controller_params': {'deltaE': params['controllers.sampler_eg.deltaE'],
                                    'convergence_level': params['controllers.sampler_eg.convergence_level']}
                },
            'Minimizer_eg':
                {'n': params['controllers.minimizer_eg.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': params['controllers.minimizer_eg.n_final'],
                                'increase_step': params['controllers.minimizer_eg.increase_step'],
                                'increase_rate': params['controllers.minimizer_eg.increase_rate']
                                },
                'controller_params': {'deltaE': params['controllers.minimizer_eg.deltaE'],
                                    'convergence_level': params['controllers.minimizer_eg.convergence_level']} #maybe at 2
                },
            'Minimizer_Samples':
                {'n': params['controllers.minimizer_samples.n'],
                'type': 'AbsDeltaEnergy',
                'change_params': {'n_final': params['controllers.minimizer_samples.n_final'],
                                'increase_step': params['controllers.minimizer_samples.increase_step'],
                                'increase_rate': params['controllers.minimizer_samples.increase_rate']
                                },
                'controller_params': {'deltaE': params['controllers.minimizer_samples.deltaE'],
                                    'convergence_level': params['controllers.minimizer_samples.convergence_level']}
                },

        }

        if  plotting_kwargs is None:
            plotting_kwargs = {}

        controllers = {key: get_controller(controller_dict, 0, False, key)
                    for key, controller_dict in controller_parameters.items()}

        # build the  likelihood
        likelihood = ift.utilities.my_sum(likelihoods.values())

        # draw position
        position = 0.1*ift.from_random(likelihood.domain)

        n_samples= lambda i: get_n_samples(sample_parameters, i, False) if i< n_global-1 \
            else params['params_inference.n_samples_posterior']


        constants = lambda i: [] if i< params['params_inference.n_single_fit'] \
            else ( ['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'])

        point_estimates = lambda i: [] if i< params['params_inference.n_single_fit'] \
            else (['log_profile_flexibility', 'log_profile_fluctuations', 'log_profile_loglogavgslope', 'log_profile_spectrum', 'log_profile_xi', 'log_profile_zeromode', 'sign_flexibility', 'sign_fluctuations', 'sign_loglogavgslope', 'sign_spectrum', 'sign_xi', 'sign_zeromode'])
        
        get_sampler = lambda i : controllers['Sampler'] if i< params['params_inference.n_single_fit'] else controllers['Sampler_eg']
    
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
            output_directory=params['params_inference.results_path'],
            resume=params['params_inference.resume']
            #dry_run=False
            )

