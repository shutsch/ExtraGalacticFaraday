
from posterior_plotter import Posterior_Plotter
from settings_maker import Settings_Maker
from src.helper_functions.parameters_maker import Parameters_maker

params = Parameters_maker().yaml_values
settings_params = Settings_Maker(params).run_settings()

plot_params = {
                'ecomponents': settings_params['ecomponents'],
                'params': params,
            }
            
Posterior_Plotter(plot_params).plot(figname='EG_posterior_custom.png')