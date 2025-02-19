import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_3():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self):
        samples = self.samples

        
        cl=np.array([s for s in samples.iterator(self.ecomponents['chi_lum'])])
        ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])

        ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        
        sl=np.sqrt(vl.val)
        si0=np.sqrt(vi0.val)
        se0=np.sqrt(ve0.val)

        print('cl', ml.val, 'pm', sl)
        print('ci0', mi0.val, 'pm', si0)
        print('ce0', me0.val, 'pm', se0)

        cl_list=[]
        ci0_list=[]
        ce0_list=[]
        for i in range(0,len(cl)):
            cl_list.append(cl[i].val)
            ci0_list.append(ci0[i].val)
            ce0_list.append(ce0[i].val)
        cl_array=np.array(cl_list)
        ci0_array=np.array(ci0_list)
        ce0_array=np.array(ce0_list)

        fig, axs = plt.subplots(2, 2)
        
        axs[0,0].scatter(cl_array, ci0_array, color='k')
        axs[0,0].set_ylabel('$\chi_{int,0}$')
        axs[0,0].set_ylim(mi0.val-5*si0,mi0.val+5*si0)
        axs[0,0].set_xlim(ml.val-5*sl,ml.val+5*sl)

        axs[0,1].scatter(ce0_array, ci0_array, color='k')
        axs[0,1].set_xlabel('$\chi_{env,0}$')
        axs[0,1].set_ylim(mi0.val-5*si0,mi0.val+5*si0)
        axs[0,1].set_xlim(me0.val-5*se0,me0.val+5*se0)

        axs[1,0].scatter(cl_array, ce0_array, color='k')
        axs[1,0].set_ylabel('$\chi_{env,0}$')
        axs[1,0].set_xlabel('$\chi_{lum}$')
        axs[1,0].set_xlim(ml.val-5*sl,ml.val+5*sl)
        axs[1,0].set_ylim(me0.val-5*se0,me0.val+5*se0)

        axs[1,1].axis('off')

        ellipse1_1sigma = Ellipse(xy=(ml.val, mi0.val), width=1*2*sl, height=1*2*si0, edgecolor='green', fc='None', lw=2)
        ellipse1_2sigma = Ellipse(xy=(ml.val, mi0.val), width=2*2*sl, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
        ellipse1_3sigma = Ellipse(xy=(ml.val, mi0.val), width=3*2*sl, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
        axs[0,0].add_patch(ellipse1_1sigma)
        axs[0,0].add_patch(ellipse1_2sigma)
        axs[0,0].add_patch(ellipse1_3sigma)


        ellipse3_1sigma = Ellipse(xy=(me0.val, mi0.val), width=1*2*se0, height=1*2*si0, edgecolor='green', fc='None', lw=2)
        ellipse3_2sigma = Ellipse(xy=(me0.val, mi0.val), width=2*2*se0, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
        ellipse3_3sigma = Ellipse(xy=(me0.val, mi0.val), width=3*2*se0, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
        axs[0,1].add_patch(ellipse3_1sigma)
        axs[0,1].add_patch(ellipse3_2sigma)
        axs[0,1].add_patch(ellipse3_3sigma)

        ellipse4_1sigma = Ellipse(xy=(ml.val, me0.val), width=1*2*sl, height=1*2*se0, edgecolor='green', fc='None', lw=2)
        ellipse4_2sigma = Ellipse(xy=(ml.val, me0.val), width=2*2*sl, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
        ellipse4_3sigma = Ellipse(xy=(ml.val, me0.val), width=3*2*sl, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
        axs[1,0].add_patch(ellipse4_1sigma)
        axs[1,0].add_patch(ellipse4_2sigma)
        axs[1,0].add_patch(ellipse4_3sigma)


        ellipse_prior1 = Ellipse(xy=(self.params['prior_mean.prior_mean_lum'], self.params['prior_mean.prior_mean_int']), width=2*self.params['prior_std.prior_std_lum'], height=2*self.params['prior_std.prior_std_int'], edgecolor='red', fc='None', lw=1)
        ellipse_prior2 = Ellipse(xy=(self.params['prior_mean.prior_mean_env'], self.params['prior_mean.prior_mean_int']), width=2*self.params['prior_std.prior_std_env'], height=2*self.params['prior_std.prior_std_int'], edgecolor='red', fc='None', lw=1)
        ellipse_prior3= Ellipse(xy=(self.params['prior_mean.prior_mean_lum'], self.params['prior_mean.prior_mean_env']), width=2*self.params['prior_std.prior_std_lum'], height=2*self.params['prior_std.prior_std_env'], edgecolor='red', fc='None', lw=1)

        axs[1,0].add_patch(ellipse_prior1)
        axs[0,1].add_patch(ellipse_prior2)
        axs[0,0].add_patch(ellipse_prior3)

        plt.subplots_adjust(wspace=0, hspace=0)

        axs[0,0].axhline(y = self.params['mean.mean_int'], color = 'b', linestyle = '--') 
        axs[0,0].axvline(x = self.params['mean.mean_lum'], color = 'b', linestyle='--')

        axs[1,0].axhline(y = self.params['mean.mean_env'], color = 'b', linestyle = '--') 
        axs[1,0].axvline(x = self.params['mean.mean_lum'], color = 'b', linestyle='--')

        axs[0,1].axhline(y = self.params['mean.mean_int'], color = 'b', linestyle = '--') 
        axs[0,1].axvline(x = self.params['mean.mean_env'], color = 'b', linestyle='--')


        axs[1,0].sharex(axs[0,0])
        axs[0,1].sharex(axs[1,1])
        axs[0,1].sharey(axs[0,0])
        axs[1,1].sharey(axs[1,0])

        axs[1,0].tick_params(labelbottom=True, direction='in')
        axs[0,1].tick_params(labelbottom=True, direction='in')
        axs[0,1].tick_params(labelleft=False, direction='in')
        axs[1,1].tick_params(labelleft=False, direction='in')
        axs[0,0].tick_params(labelleft=True, labelbottom=False, direction='in')
