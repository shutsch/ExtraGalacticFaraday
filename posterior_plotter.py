import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib

from single_plotters.plot_posterior_1 import Posterior_Plotter_1
from single_plotters.plot_posterior_2 import Posterior_Plotter_2
from single_plotters.plot_posterior_22 import Posterior_Plotter_22
from single_plotters.plot_posterior_3 import Posterior_Plotter_3
from single_plotters.plot_posterior_4 import Posterior_Plotter_4
matplotlib.use('TkAgg')


class Posterior_Plotter():
    def __init__(self, args):
        self.args = args
        self.params = args['params']

    def plot(self, figname):

        samples = ift.ResidualSampleList.load(f'{self.params["params_inference.results_path"]}pickle/last')

        if self.params['params_mock_cat.maker_params.n_eg_params'] == 1:
            Posterior_Plotter_1(samples=samples, args=self.args).plot()
        elif self.params['params_mock_cat.maker_params.n_eg_params'] == 2:
            Posterior_Plotter_2(samples=samples, args=self.args).plot()
        elif self.params['params_mock_cat.maker_params.n_eg_params'] == 22:
            Posterior_Plotter_22(samples=samples, args=self.args).plot()
        elif self.params['params_mock_cat.maker_params.n_eg_params'] == 3:
            Posterior_Plotter_3(samples=samples, args=self.args).plot()
        elif self.params['params_mock_cat.maker_params.n_eg_params'] == 4:
            Posterior_Plotter_4(samples=samples, args=self.args).plot()
    
        plt.savefig(f'{self.params["params_inference.plot_path"]}{figname}', bbox_inches='tight')
    

        


