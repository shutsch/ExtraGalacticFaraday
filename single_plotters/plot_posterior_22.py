import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
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
        me0, ve0 = samples.sample_stat(self.ecomponents['env'])

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

        fig, axs = plt.subplots(1, 1)
        

        axs.scatter(cr_array, ce0_array, color='k')
        axs.set_xlabel('$\chi_{red}$')
        axs.set_ylabel('$\chi_{env,0}$')
        axs.set_xlim(mr.val-5*sr,mr.val+5*sr)
        axs.set_ylim(me0.val-5*se0,me0.val+5*se0)

        ellipse_1sigma = Ellipse(xy=(mr.val, me0.val), width=1*2*sr, height=1*2*se0, edgecolor='green', fc='None', lw=2)
        ellipse_2sigma = Ellipse(xy=(mr.val, me0.val), width=2*2*sr, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
        ellipse_3sigma = Ellipse(xy=(mr.val, me0.val), width=3*2*sr, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
        axs.add_patch(ellipse_1sigma)
        axs.add_patch(ellipse_2sigma)
        axs.add_patch(ellipse_3sigma)




        ellipse_prior1 = Ellipse(xy=(self.params['prior_mean.prior_mean_red'], self.params['prior_mean.prior_mean_env']), width=2*self.params['prior_std.prior_std_red'], height=2*self.params['prior_std.prior_std_env'], edgecolor='red', fc='None', lw=1)

        axs.add_patch(ellipse_prior1)


        plt.subplots_adjust(wspace=0, hspace=0)



        axs.axhline(y = self.params['mean.mean_env'], color = 'b', linestyle = '--') 
        axs.axvline(x = self.params['mean.mean_red'], color = 'b', linestyle='--')
        axs.tick_params(labelbottom=True, labelleft=True, direction='in')
