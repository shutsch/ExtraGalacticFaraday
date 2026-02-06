
from eta_plotter import Eta_Plotter
from settings_maker import Settings_Maker
from src.helper_functions.parameters_maker import Parameters_maker

params = Parameters_maker().yaml_values
settings_params = Settings_Maker(params).run_settings()

plot_params = {
                'emodel': settings_params['emodel'],
                'ecomponents': settings_params['ecomponents'],
                'params': params,
            }
            
Eta_Plotter(plot_params).plot('noise_excitations',params['params_inference.plot_path'], string=params['run.name'])
