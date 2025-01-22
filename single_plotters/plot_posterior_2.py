import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
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

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        
        si0=np.sqrt(vi0.val)
        se0=np.sqrt(ve0.val)

        print('ci0', mi0.val, 'pm', si0)
        print('ce0', me0.val, 'pm', se0)

        ci0_list=[]
        ce0_list=[]
        for i in range(0,len(ci0)):
            ci0_list.append(ci0[i].val)
            ce0_list.append(ce0[i].val)
        ci0_array=np.array(ci0_list)
        ce0_array=np.array(ce0_list)

        fig, axs = plt.subplots(1, 1)
        

        axs.scatter(ce0_array, ci0_array, color='k')
        axs.set_xlabel('$\chi_{env,0}$')
        axs.set_ylabel('$\chi_{int,0}$')
        axs.set_xlim(me0.val-5*se0,me0.val+5*se0)
        axs.set_ylim(mi0.val-5*si0,mi0.val+5*si0)

        ellipse_1sigma = Ellipse(xy=(me0.val, mi0.val), width=1*2*se0, height=1*2*si0, edgecolor='green', fc='None', lw=2)
        ellipse_2sigma = Ellipse(xy=(me0.val, mi0.val), width=2*2*se0, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
        ellipse_3sigma = Ellipse(xy=(me0.val, mi0.val), width=3*2*se0, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
        axs.add_patch(ellipse_1sigma)
        axs.add_patch(ellipse_2sigma)
        axs.add_patch(ellipse_3sigma)




        ellipse_prior1 = Ellipse(xy=(self.params['prior_mean.prior_mean_env'], self.params['prior_mean.prior_mean_int']), width=2*self.params['prior_std.prior_std_env'], height=2*self.params['prior_std.prior_std_int'], edgecolor='red', fc='None', lw=1)

        axs.add_patch(ellipse_prior1)


        plt.subplots_adjust(wspace=0, hspace=0)



        axs.axhline(y = self.params['mean.mean_int'], color = 'b', linestyle = '--') 
        axs.axvline(x = self.params['mean.mean_env'], color = 'b', linestyle='--')