import numpy as np
from src.helper_functions.plot.plot import _density_estimation
from astropy.modeling.models import Gaussian1D
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_22():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self):
        samples = self.samples

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])

        cr=np.array([s for s in samples.iterator(self.ecomponents['chi_red'])])
        mr, vr = samples.sample_stat(self.ecomponents['chi_red'])
        
        se0=np.sqrt(ve0.val)
        sr=np.sqrt(vr.val)

        print('ce0', me0.val, 'pm', se0)
        print('cr', mr.val, 'pm', sr)

        ce0_list=[]
        cr_list=[]
        for i in range(0,len(ce0)):
            ce0_list.append(ce0[i].val)
            cr_list.append(cr[i].val)
        ce0_array=np.array(ce0_list)
        cr_array=np.array(cr_list)

        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        
        plt.subplots_adjust(wspace=0, hspace=0)

        xxx, yyy, zzz = _density_estimation(cr_array, ce0_array, mr.val-5*sr,mr.val+5*sr, me0.val-5*se0,me0.val+5*se0, 100)
        axs[0,0].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[mr.val-5*sr,mr.val+5*sr, me0.val-5*se0,me0.val+5*se0], aspect="auto")
        axs[0,0].scatter(cr_array, ce0_array, color='k', s=self.params['plot.markersize'])
        axs[0,0].set_xlabel('$\\chi_{red}$', fontsize = self.params['plot.fontsize'])
        axs[0,0].set_ylabel('$\\chi_{env,0}$', fontsize = self.params['plot.fontsize'])
        axs[0,0].set_xlim(mr.val-5*sr,mr.val+5*sr)
        axs[0,0].set_ylim(me0.val-5*se0,me0.val+5*se0)





        axs[0,0].axhline(y = self.params['mean.mean_int'], color = 'k', linestyle = '-') 
        axs[0,0].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle='-')
        axs[0,0].tick_params(labelbottom=True, labelleft=False, direction='in')



        axs[1,0].hist(cr_array, bins=self.params['plot.bins'], color='lightgray')
        axs[1,0].set_xlabel('$\\chi_{red}$', fontsize = self.params['plot.fontsize'])
        axs[1,0].tick_params('y', labelleft=False)
        axs[1,0].set_xlim(mr.val-5*sr,mr.val+5*sr)

        axs[1,0].axvline(x = mr.val+sr, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[1,0].axvline(x = mr.val-sr, color = 'green', linestyle='--')

        axs[1,0].axvline(x = mr.val+2*sr, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[1,0].axvline(x = mr.val-2*sr, color = 'orange', linestyle='--')

        axs[1,0].axvline(x = mr.val+3*sr, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[1,0].axvline(x = mr.val-3*sr, color = 'red', linestyle='--')

        axs[1,0].axvline(x = self.params['mean.mean_red'], color = 'k', linestyle = '-', label='Mock') 
       


        x = np.linspace(mr.val-5*sr,mr.val+5*sr, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_red'], stddev= self.params['prior_std.prior_std_red'])
      
        axs[1,0].plot(x, y(x), 'b-', label='Prior')



        axs[0,1].hist(ce0_array, bins=self.params['plot.bins'], color='lightgray')
        axs[0,1].set_xlabel('$\\chi_{env, 0}$', fontsize = self.params['plot.fontsize'])
        axs[0,1].tick_params('y', labelleft=False)
        axs[0,1].set_xlim(me0.val-5*se0,me0.val+5*se0)

        axs[0,1].axvline(x = me0.val+se0, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[0,1].axvline(x = me0.val-se0, color = 'green', linestyle='--')
        
        axs[0,1].axvline(x = me0.val+2*se0, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[0,1].axvline(x = me0.val-2*se0, color = 'orange', linestyle='--')

        axs[0,1].axvline(x = me0.val+3*se0, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[0,1].axvline(x = me0.val-3*se0, color = 'red', linestyle='--')

        axs[0,1].axvline(x = self.params['mean.mean_env'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(me0.val-5*se0,me0.val+5*se0, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_env'], stddev= self.params['prior_std.prior_std_env'])
      
        axs[0,1].plot(x, y(x), 'b-', label='Prior')

        axs[1,1].axis('off')
        axs[0,2].axis('off')
        axs[0,3].axis('off')
        axs[1,2].axis('off')
        axs[1,3].axis('off')
        axs[2,0].axis('off')
        axs[2,1].axis('off')
        axs[2,2].axis('off')
        axs[2,3].axis('off')
        axs[3,0].axis('off')
        axs[3,1].axis('off')
        axs[3,2].axis('off')
        axs[3,3].axis('off')
        
        lines = []
        labels = []
        Line, Label = axs[0,1].get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)
        fig.legend(lines, labels, bbox_to_anchor=(0.05, 0.888), fontsize = self.params['plot.legend_fontsize'])
