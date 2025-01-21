import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_3():
    def __init__(self, samples, args):

        self.ecomponents = args['ecomponents']
        self.n_params= args['n_eg_params']
        self.results_path=args['results_path']
        self.plot_path=args['plot_path']
        self.samples = samples

    def plot(self):
        samples = self.samples

        
        cr=np.array([s for s in samples.iterator(self.ecomponents['chi_red'])])
        mr, vr = samples.sample_stat(self.ecomponents['chi_red'])

        ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        
        sr=np.sqrt(vr.val)
        si0=np.sqrt(vi0.val)
        se0=np.sqrt(ve0.val)

        print('cr', mr.val, 'pm', sr)
        print('ci0', mi0.val, 'pm', si0)
        print('ce0', me0.val, 'pm', se0)

        cr_list=[]
        ci0_list=[]
        ce0_list=[]
        for i in range(0,len(cr)):
            cr_list.append(cr[i].val)
            ci0_list.append(ci0[i].val)
            ce0_list.append(ce0[i].val)
        cr_array=np.array(cr_list)
        ci0_array=np.array(ci0_list)
        ce0_array=np.array(ce0_list)

        fig, axs = plt.subplots(2, 2)
        
        axs[0,0].scatter(cr_array, ci0_array, color='k')
        axs[0,0].set_ylabel('$\chi_{int,0}$')
        axs[0,0].set_ylim(-3,8)

        axs[0,1].scatter(ce0_array, ci0_array, color='k')
        axs[0,1].set_xlabel('$\chi_{env,0}$')
        axs[0,1].set_yticklabels([])
        axs[0,1].set_ylim(-3,8)

        axs[1,0].scatter(cr_array, ce0_array, color='k')
        axs[1,0].set_ylabel('$\chi_{env,0}$')
        axs[1,0].set_xlabel('$\chi_{red}$')

        #axs[0,1].axis('off')


        #axs[1,0].axis('off')
        axs[1,1].axis('off')

        ellipse1_1sigma = Ellipse(xy=(mr.val, mi0.val), width=1*2*sr, height=1*2*si0, edgecolor='green', fc='None', lw=2)
        ellipse1_2sigma = Ellipse(xy=(mr.val, mi0.val), width=2*2*sr, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
        ellipse1_3sigma = Ellipse(xy=(mr.val, mi0.val), width=3*2*sr, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
        axs[0,0].add_patch(ellipse1_1sigma)
        axs[0,0].add_patch(ellipse1_2sigma)
        axs[0,0].add_patch(ellipse1_3sigma)


        ellipse3_1sigma = Ellipse(xy=(me0.val, mi0.val), width=1*2*se0, height=1*2*si0, edgecolor='green', fc='None', lw=2)
        ellipse3_2sigma = Ellipse(xy=(me0.val, mi0.val), width=2*2*se0, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
        ellipse3_3sigma = Ellipse(xy=(me0.val, mi0.val), width=3*2*se0, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
        axs[0,1].add_patch(ellipse3_1sigma)
        axs[0,1].add_patch(ellipse3_2sigma)
        axs[0,1].add_patch(ellipse3_3sigma)

        ellipse4_1sigma = Ellipse(xy=(mr.val, me0.val), width=1*2*sr, height=1*2*se0, edgecolor='green', fc='None', lw=2)
        ellipse4_2sigma = Ellipse(xy=(mr.val, me0.val), width=2*2*sr, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
        ellipse4_3sigma = Ellipse(xy=(mr.val, me0.val), width=3*2*sr, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
        axs[1,0].add_patch(ellipse4_1sigma)
        axs[1,0].add_patch(ellipse4_2sigma)
        axs[1,0].add_patch(ellipse4_3sigma)


        ellipse_prior1 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
        ellipse_prior2 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
        ellipse_prior3= Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)

        axs[1,0].add_patch(ellipse_prior1)
        axs[0,1].add_patch(ellipse_prior2)
        axs[0,0].add_patch(ellipse_prior3)

        plt.subplots_adjust(wspace=0, hspace=0)

        axs[0,0].axhline(y = 5.0, color = 'b', linestyle = '--') 
        axs[0,0].axvline(x = -0.5, color = 'b', linestyle='--')

        axs[1,0].axhline(y = 0.0, color = 'b', linestyle = '--') 
        axs[1,0].axvline(x = -0.5, color = 'b', linestyle='--')

        axs[0,1].axhline(y = 5.0, color = 'b', linestyle = '--') 
        axs[0,1].axvline(x = 0.0, color = 'b', linestyle='--')