import nifty8 as ift
import numpy as np
import libs as Egf
from src.helper_functions.plot.plot import _density_estimation
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')




class Posterior_Plotter():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self, figname):
        #samples = ift.ResidualSampleList.load(f'{self.params["file_params.results_path"]}pickle/last')
        stats=Egf.sample_statistics(self, self.samples)

        #stats = self.samples.sample_statistics()
        mr = stats['chi_red_mean']
        sr = stats['chi_red_std']   
        cr = stats['chi_red_samples'] 

        ml = stats['chi_lum_mean']
        sl = stats['chi_lum_std']
        cl = stats['chi_lum_samples']

        mi0 = stats['chi_int_0_mean']
        si0 = stats['chi_int_0_std']
        ci0 = stats['chi_int_0_samples']

        me0 = stats['chi_env_0_mean']
        se0 = stats['chi_env_0_std']
        ce0 = stats['chi_env_0_samples']

        width = 5.0
        points = 100

        fig, axs = plt.subplots(4, 4, figsize=(15, 15)) #, layout="constrained"

        plt.subplots_adjust(wspace=0, hspace=0)

        Egf.density_plot(self.params,axs,cr,ci0,mr,mi0,sr,si0, width, points, 0, 0, xlabel=None, ylabel='$\\chi_{int,0}$')
        Egf.density_plot(self.params,axs,cl,ci0,ml,mi0,sl,si0, width, points, 0, 1)
        Egf.density_plot(self.params,axs,ce0,ci0,me0,mi0,se0,si0, width, points, 0, 2, xlabel='$\\chi_{env,0}$')
        Egf.density_plot(self.params,axs,cr,ce0,mr,me0,sr,se0, width, points, 1, 0, xlabel=None, ylabel='$\\chi_{env,0}$')
        Egf.density_plot(self.params,axs,cl,ce0,ml,me0,sl,se0, width, points, 1, 1, xlabel='$\\chi_{lum}$')
        Egf.density_plot(self.params,axs,cr,cl,mr,ml,sr,sl, width, points, 2, 0, xlabel='$\\chi_{red}$', ylabel='$\\chi_{lum}$')
       


        axs[0,0].axhline(y = self.params['mean.mean_int'], color = 'k', linestyle = '-') 
        axs[0,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle='-')

        axs[1,0].axhline(y = self.params['mean.mean_env'], color = 'k', linestyle = '-') 
        axs[1,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle='-')

        axs[2,0].axhline(y = self.params['mean.mean_lum'], color = 'k', linestyle = '-') 
        axs[2,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle='-')

        axs[0,1].axhline(y = self.params['mean.mean_int'], color = 'k', linestyle = '-') 
        axs[0,1].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle='-')

        axs[1,1].axhline(y = self.params['mean.mean_env'], color = 'k', linestyle = '-') 
        axs[1,1].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle='-')

        axs[0,2].axhline(y = self.params['mean.mean_int'], color = 'k', linestyle = '-') 
        axs[0,2].axvline(x = self.params['mean.mean_env'], color = 'k', linestyle='-')

        axs[1,0].sharex(axs[0,0])
        axs[2,0].sharex(axs[0,0])
        axs[0,1].sharex(axs[1,1])
        axs[0,1].sharey(axs[0,0])
        axs[0,2].sharey(axs[0,0])
        axs[1,1].sharey(axs[1,0])


        Egf.histo_plot(self.params, axs,cr, mr, sr, width, 3, 0, xlabel='$\\chi_{red}$')
        Egf.sigma_plot(self.params, axs, mr, sr, np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 3, 0)
        axs[3,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle = '-', label='Mock') 
        #amplitude parameter might need to be adjusted       
        Egf.gauss_plot(self.params, axs, mr, sr, width, 3, 0, 1000, label='Prior')

        Egf.histo_plot(self.params, axs,cl, ml, sl, width, 2, 1, xlabel='$\\chi_{lum}$')
        Egf.sigma_plot(self.params, axs, ml, sl, np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 2, 1)
        axs[2,1].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle = '-', label='Mock') 
        Egf.gauss_plot(self.params, axs, ml, sl, width, 2, 1, 1000, label='Prior')

        Egf.histo_plot(self.params, axs,ce0, me0, se0, width, 1, 2, xlabel='$\\chi_{env, 0}$')
        Egf.sigma_plot(self.params, axs, me0, se0, np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 1, 2)
        axs[1,2].axvline(x = self.params['mean.mean_env'], color = 'k', linestyle = '-', label='Mock') 
        Egf.gauss_plot(self.params, axs, me0, se0, width, 1, 2, 1000, label='Prior')



        Egf.histo_plot(self.params, axs,ci0, mi0, si0, width, 0, 3, xlabel='$\\chi_{int, 0}$')
        Egf.sigma_plot(self.params, axs, mi0, si0, np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 0, 3)
        axs[0,3].axvline(x = self.params['mean.mean_int'], color = 'k', linestyle = '-', label='Mock') 
        Egf.gauss_plot(self.params, axs, mi0, si0, width, 0, 3, 1000, label='Prior')


        axs[1,0].tick_params(labelbottom=True, direction='in')
        axs[0,1].tick_params(labelbottom=True, direction='in')
        axs[0,1].tick_params(labelleft=False, direction='in')
        axs[0,2].tick_params(labelleft=False, direction='in')
        axs[1,1].tick_params(labelleft=False, direction='in')
        axs[0,0].tick_params(labelbottom=True, direction='in')
        axs[2,0].tick_params(labelbottom=True, labelleft=True, direction='in')


        axs[2,2].axis('off')
        axs[3,1].axis('off')
        axs[3,2].axis('off')
        axs[3,3].axis('off')
        axs[2,3].axis('off')
        axs[1,3].axis('off')

        lines = []
        labels = []
        Line, Label = axs[0,3].get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)
        fig.legend(lines, labels, bbox_to_anchor=(0.05, 0.888), fontsize = self.params['plot.legend_fontsize'])

        plt.savefig(f'{self.params["file_params.plot_path"]}{figname}', bbox_inches='tight')
