import numpy as np
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_1():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self):
        samples = self.samples

        c1=np.array([s for s in samples.iterator(self.ecomponents['chi1'])])
        m1, v1 = samples.sample_stat(self.ecomponents['chi1'])
        s1=np.sqrt(v1.val)
        print('chi1', m1.val, s1)

        c1_list=[]
        for i in range(0,len(c1)):
            c1_list.append(c1[i].val)
        c1_array=np.array(c1_list)

        fig, axs = plt.subplots(4,4, figsize=(15, 15))

        plt.subplots_adjust(wspace=0, hspace=0)

        axs[0,0].hist(c1_array, bins=self.params['plot.bins'], color='lightgray')
        #ax.set_ylabel('#')
        axs[0,0].set_xlabel('$\\chi_{1}$', fontsize = self.params['plot.fontsize'])

        axs[0,0].axvline(x = m1.val+s1, color = 'green', linestyle='--', label='1-$\\sigma$')
        axs[0,0].axvline(x = m1.val-s1, color = 'green', linestyle='--')

        axs[0,0].axvline(x = m1.val+2*s1, color = 'orange', linestyle='--', label='2-$\\sigma$')
        axs[0,0].axvline(x = m1.val-2*s1, color = 'orange', linestyle='--')

        axs[0,0].axvline(x = m1.val+3*s1, color = 'red', linestyle='--', label='3-$\\sigma$')
        axs[0,0].axvline(x = m1.val-3*s1, color = 'red', linestyle='--')

        axs[0,0].axvline(x = self.params['mean.mean_one'], color = 'k', linestyle = '-', label='Mock') 
       

        x = np.linspace(m1.val-5*s1,m1.val+5*s1, 1000)
        #amplitude might need to be adjusted
        y = Gaussian1D(amplitude=self.params['plot.amplitude'], mean=self.params['prior_mean.prior_mean_one'], stddev= self.params['prior_std.prior_std_one'])

        axs[0,0].plot(x, y(x), 'b-', label='Prior')
        
        axs[0,0].tick_params('y', labelleft=False)

        axs[0,0].set_xlim(m1.val-5*s1,m1.val+5*s1)
        
        axs[0,0].tick_params(axis='y',direction='in')
        axs[0,0].tick_params(axis='x',direction='in')
        

        
        axs[0,1].axis('off')
        axs[0,2].axis('off')
        axs[0,3].axis('off')
        axs[1,0].axis('off')
        axs[1,1].axis('off')
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
        Line, Label = axs[0,0].get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)
        fig.legend(lines, labels, bbox_to_anchor=(0.05, 0.888), fontsize = self.params['plot.legend_fontsize'])
