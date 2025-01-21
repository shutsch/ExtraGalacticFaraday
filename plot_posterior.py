import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib

from single_plotters.plot_posterior_1 import Posterior_Plotter_1
from single_plotters.plot_posterior_2 import Posterior_Plotter_2
from single_plotters.plot_posterior_3 import Posterior_Plotter_3
from single_plotters.plot_posterior_4 import Posterior_Plotter_4
matplotlib.use('TkAgg')


class Posterior_Plotter():
    def __init__(self, args):

        self.args = args
        self.n_params= args['n_eg_params']
        self.results_path=args['results_path']
        self.plot_path=args['plot_path']

    def plot(self):

        samples = ift.ResidualSampleList.load(self.results_path + 'pickle/last')

        if self.n_params == 1:
            Posterior_Plotter_1(samples=samples, args=self.args).plot()
        elif self.n_params == 2:
            Posterior_Plotter_2(samples=samples, args=self.args).plot()
        elif self.n_params == 3:
            Posterior_Plotter_3(samples=samples, args=self.args).plot()
        elif self.n_params == 4:
            Posterior_Plotter_4(samples=samples, args=self.args).plot()
    
        plt.savefig(f'{self.plot_path}EG_posterior.png', bbox_inches='tight')
    

        


