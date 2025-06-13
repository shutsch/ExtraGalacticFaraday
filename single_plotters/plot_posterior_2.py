import numpy as np
from src.helper_functions.plot.plot import _density_estimation
from astropy.modeling.models import Gaussian1D
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_2():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self):
        samples = self.samples

        ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])

        cl=np.array([s for s in samples.iterator(self.ecomponents['chi_lum'])])
        ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])
        
        si0=np.sqrt(vi0.val)
        sl=np.sqrt(vl.val)

        print('ci0', mi0.val, 'pm', si0)
        print('cl', ml.val, 'pm', sl)

        ci0_list=[]
        cl_list=[]
        for i in range(0,len(ci0)):
            ci0_list.append(ci0[i].val)
            cl_list.append(cl[i].val)
        ci0_array=np.array(ci0_list)
        cl_array=np.array(cl_list)

        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        plt.subplots_adjust(wspace=0, hspace=0)


        xxx, yyy, zzz = _density_estimation(cl_array, ci0_array, ml.val-5*sl,ml.val+5*sl, mi0.val-5*si0,mi0.val+5*si0, 100)
        axs[0,0].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[ml.val-5*sl,ml.val+5*sl, mi0.val-5*si0,mi0.val+5*si0], aspect="auto")
        axs[0,0].scatter(cl_array, ci0_array, color='k', s=self.params['plot.markersize'])
        axs[0,0].set_xlabel('$\\chi_{lum}$', fontsize = self.params['plot.fontsize'])
        axs[0,0].set_ylabel('$\\chi_{int,0}$', fontsize = self.params['plot.fontsize'])
        axs[0,0].set_xlim(ml.val-5*sl,ml.val+5*sl)
        axs[0,0].set_ylim(mi0.val-5*si0,mi0.val+5*si0)






        axs[0,0].axhline(y = self.params['mean.mean_int'], color = 'k', linestyle = '-') 
        axs[0,0].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle='-')
        axs[0,0].tick_params(labelbottom=True, labelleft=False, direction='in')



        axs[1,0].hist(cl_array, bins=self.params['plot.bins'], color='lightgray')
        axs[1,0].set_xlabel('$\\chi_{lum}$', fontsize = self.params['plot.fontsize'])
        axs[1,0].tick_params('y', labelleft=False)
        axs[1,0].set_xlim(ml.val-5*sl,ml.val+5*sl)

        axs[1,0].axvline(x = ml.val+sl, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[1,0].axvline(x = ml.val-sl, color = 'green', linestyle='--')

        axs[1,0].axvline(x = ml.val+2*sl, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[1,0].axvline(x = ml.val-2*sl, color = 'orange', linestyle='--')

        axs[1,0].axvline(x = ml.val+3*sl, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[1,0].axvline(x = ml.val-3*sl, color = 'red', linestyle='--')

        axs[1,0].axvline(x = self.params['mean.mean_lum'], color = 'k', linestyle = '-', label='Mock') 
       


        x = np.linspace(ml.val-5*sl,ml.val+5*sl, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_lum'], stddev= self.params['prior_std.prior_std_lum'])
      
        axs[1,0].plot(x, y(x), 'b-', label='Prior')



        axs[0,1].hist(ci0_array, bins=self.params['plot.bins'], color='lightgray')
        axs[0,1].set_xlabel('$\\chi_{int}$', fontsize = self.params['plot.fontsize'])
        axs[0,1].tick_params('y', labelleft=False)
        axs[0,1].set_xlim(mi0.val-5*si0,mi0.val+5*si0)

        axs[0,1].axvline(x = mi0.val+si0, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[0,1].axvline(x = mi0.val-si0, color = 'green', linestyle='--')

        axs[0,1].axvline(x = mi0.val+2*si0, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[0,1].axvline(x = mi0.val-2*si0, color = 'orange', linestyle='--')

        axs[0,1].axvline(x = mi0.val+3*si0, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[0,1].axvline(x = mi0.val-3*si0, color = 'red', linestyle='--')

        axs[0,1].axvline(x = self.params['mean.mean_int'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(mi0.val-5*si0,mi0.val+5*si0, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_int'], stddev= self.params['prior_std.prior_std_int'])
      
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
