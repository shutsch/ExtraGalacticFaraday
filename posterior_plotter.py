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

        fig, axs = plt.subplots(4, 4, figsize=(15, 15)) #, layout="constrained"

        plt.subplots_adjust(wspace=0, hspace=0)

        xxx, yyy, zzz = _density_estimation(cr, ci0, mr-width*sr,mr+width*sr, mi0-width*si0,mi0+width*si0, 100)
        axs[0,0].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[mr-width*sr,mr+width*sr, mi0-width*si0,mi0+width*si0], aspect="auto")
        axs[0,0].scatter(cr, ci0, color='k', s=self.params['plot.markersize'])
        axs[0,0].set_ylabel('$\\chi_{int,0}$', fontsize = self.params['plot.fontsize'])
        axs[0,0].set_ylim(mi0-width*si0,mi0+width*si0)
        axs[0,0].set_xlim(mr-width*sr,mr+width*sr)



        xxx, yyy, zzz = _density_estimation(cl, ci0, ml-width*sl,ml+width*sl, mi0-width*si0,mi0+width*si0, 100)
        axs[0,1].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[ml-width*sl,ml+width*sl, mi0-width*si0,mi0+width*si0], aspect="auto")
        axs[0,1].scatter(cl, ci0, color='k', s=self.params['plot.markersize'])
        axs[0,1].set_ylim(mi0-width*si0,mi0+width*si0)
        axs[0,1].set_xlim(ml-width*sl,ml+width*sl)

        xxx, yyy, zzz = _density_estimation(ce0, ci0, me0-width*se0,me0+width*se0, mi0-width*si0,mi0+width*si0, 100)
        axs[0,2].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[me0-width*se0,me0+width*se0, mi0-width*si0,mi0+width*si0], aspect="auto")
        axs[0,2].scatter(ce0, ci0, color='k', s=self.params['plot.markersize'])
        axs[0,2].set_xlabel('$\\chi_{env,0}$', fontsize = self.params['plot.fontsize'])
        axs[0,2].set_ylim(mi0-width*si0,mi0+width*si0)
        axs[0,2].set_xlim(me0-width*se0,me0+width*se0)


        xxx, yyy, zzz = _density_estimation(cr, ce0, mr-width*sr,mr+width*sr, me0-width*se0,me0+width*se0, 100)
        axs[1,0].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[mr-width*sr,mr+width*sr, me0-width*se0,me0+width*se0], aspect="auto")
        axs[1,0].scatter(cr, ce0, color='k', s=self.params['plot.markersize'])
        axs[1,0].set_ylabel('$\\chi_{env,0}$', fontsize = self.params['plot.fontsize'])
        axs[1,0].set_xlim(mr-width*sr,mr+width*sr)
        axs[1,0].set_ylim(me0-width*se0,me0+width*se0)


        xxx, yyy, zzz = _density_estimation(cl, ce0, ml-width*sl,ml+width*sl, me0-width*se0,me0+width*se0, 100)
        axs[1,1].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[ml-width*sl,ml+width*sl, me0-width*se0,me0+width*se0], aspect="auto")
        axs[1,1].scatter(cl, ce0, color='k', s=self.params['plot.markersize'])
        axs[1,1].set_xlabel('$\\chi_{lum}$', fontsize = self.params['plot.fontsize'])
        axs[1,1].set_xlim(ml-width*sl,ml+width*sl)
        axs[1,1].set_ylim(me0-width*se0,me0+width*se0)


        xxx, yyy, zzz = _density_estimation(cr, cl, mr-width*sr,mr+width*sr, ml-width*sl,ml+width*sl, 100)
        axs[2,0].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[mr-width*sr,mr+width*sr, ml-width*sl,ml+width*sl], aspect="auto")
        axs[2,0].scatter(cr, cl, color='k', s=self.params['plot.fontsize'])
        axs[2,0].set_xlabel('$\\chi_{red}$', fontsize = self.params['plot.fontsize'])
        axs[2,0].set_ylabel('$\\chi_{lum}$', fontsize = self.params['plot.fontsize'])
        axs[2,0].set_xlim(mr-width*sr,mr+width*sr)
        axs[2,0].set_ylim(ml-width*sl,ml+width*sl)
       


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


        axs[3,0].hist(cr, bins=self.params['plot.bins'], color='lightgray')
        axs[3,0].set_xlabel('$\\chi_{red}$', fontsize = self.params['plot.fontsize'])
        axs[3,0].tick_params('y', labelleft=False)
        axs[3,0].set_xlim(mr-width*sr,mr+width*sr)

        axs[3,0].axvline(x = mr+sr, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[3,0].axvline(x = mr-sr, color = 'green', linestyle='--')

        axs[3,0].axvline(x = mr+2*sr, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[3,0].axvline(x = mr-2*sr, color = 'orange', linestyle='--')

        axs[3,0].axvline(x = mr+3*sr, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[3,0].axvline(x = mr-3*sr, color = 'red', linestyle='--')

        axs[3,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(mr-width*sr,mr+width*sr, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_red'], stddev= self.params['prior_std.prior_std_red'])
      
        axs[3,0].plot(x, y(x), 'b-', label='Prior')




        axs[2,1].hist(cl, bins=self.params['plot.bins'], color='lightgray')
        axs[2,1].set_xlabel('$\\chi_{lum}$', fontsize = self.params['plot.fontsize'])
        axs[2,1].tick_params('y', labelleft=False)
        axs[2,1].set_xlim(ml-width*sl,ml+width*sl)

        axs[2,1].axvline(x = ml+sl, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[2,1].axvline(x = ml-sl, color = 'green', linestyle='--')

        axs[2,1].axvline(x = ml+2*sl, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[2,1].axvline(x = ml-2*sl, color = 'orange', linestyle='--')

        axs[2,1].axvline(x = ml+3*sl, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[2,1].axvline(x = ml-3*sl, color = 'red', linestyle='--')

        axs[2,1].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(ml-width*sl,ml+width*sl, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_lum'], stddev= self.params['prior_std.prior_std_lum'])
      
        axs[2,1].plot(x, y(x), 'b-', label='Prior')







        axs[1,2].hist(ce0, bins=self.params['plot.bins'], color='lightgray')
        axs[1,2].set_xlabel('$\\chi_{env, 0}$', fontsize = self.params['plot.fontsize'])
        axs[1,2].tick_params('y', labelleft=False)
        axs[1,2].set_xlim(me0-width*se0,me0+width*se0)

        axs[1,2].axvline(x = me0+se0, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[1,2].axvline(x = me0-se0, color = 'green', linestyle='--')

        axs[1,2].axvline(x = me0+2*se0, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[1,2].axvline(x = me0-2*se0, color = 'orange', linestyle='--')

        axs[1,2].axvline(x = me0+3*se0, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[1,2].axvline(x = me0-3*se0, color = 'red', linestyle='--')

        axs[1,2].axvline(x = self.params['mean.mean_env'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(me0-width*se0,me0+width*se0, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_env'], stddev= self.params['prior_std.prior_std_env'])
      
        axs[1,2].plot(x, y(x), 'b-', label='Prior')



        axs[0,3].hist(ci0, bins=self.params['plot.bins'], color='lightgray')
        axs[0,3].set_xlabel('$\\chi_{int, 0}$', fontsize = self.params['plot.fontsize'])
        axs[0,3].tick_params('y', labelleft=False)
        axs[0,3].set_xlim(mi0-width*si0,mi0+width*si0)

        axs[0,3].axvline(x = mi0+si0, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[0,3].axvline(x = mi0-si0, color = 'green', linestyle='--')

        axs[0,3].axvline(x = mi0+2*si0, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[0,3].axvline(x = mi0-2*si0, color = 'orange', linestyle='--')

        axs[0,3].axvline(x = mi0+3*si0, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[0,3].axvline(x = mi0-3*si0, color = 'red', linestyle='--')

        axs[0,3].axvline(x = self.params['mean.mean_int'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(mi0-width*si0,mi0+width*si0, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_int'], stddev= self.params['prior_std.prior_std_int'])
      
        axs[0,3].plot(x, y(x), 'b-', label='Prior')

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
