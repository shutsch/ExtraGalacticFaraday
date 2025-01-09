import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter():
    def __init__(self, args):

        self.ecomponents = args['ecomponents']
        self.n_params= args['n_eg_params']
        self.results_path=args['results_path']
        self.plot_path=args['plot_path']

    def plot(self):
    
        if(self.n_params >= 4):

            samples = ift.ResidualSampleList.load(self.results_path + 'pickle/last')

            cr=np.array([s for s in samples.iterator(self.ecomponents['chi_red'])])
            mr, vr = samples.sample_stat(self.ecomponents['chi_red'])

            ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
            mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])

            cl=np.array([s for s in samples.iterator(self.ecomponents['chi_lum'])])
            ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])

            ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
            me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
            
            sr=np.sqrt(vr.val)
            si0=np.sqrt(vi0.val)
            sl=np.sqrt(vl.val)
            se0=np.sqrt(ve0.val)

            print('cr', mr.val, 'pm', sr)
            print('ci0', mi0.val, 'pm', si0)
            print('cl', ml.val, 'pm', sl)
            print('ce0', me0.val, 'pm', se0)




            cr_list=[]
            cl_list=[]
            ci0_list=[]
            ce0_list=[]
            for i in range(0,len(cr)):
                cr_list.append(cr[i].val)
                cl_list.append(cl[i].val)
                ci0_list.append(ci0[i].val)
                ce0_list.append(ce0[i].val)
            cr_array=np.array(cr_list)
            cl_array=np.array(cl_list)
            ci0_array=np.array(ci0_list)
            ce0_array=np.array(ce0_list)

 
            fig, axs = plt.subplots(3, 3)
            
            axs[0,0].scatter(cr_array, ci0_array, color='k')
            axs[0,0].set_ylabel('$\chi_{int,0}$')


            axs[0,1].scatter(cl_array, ci0_array, color='k')
            axs[0,1].set_yticklabels([])

            axs[0,2].scatter(ce0_array, ci0_array, color='k')
            axs[0,2].set_xlabel('$\chi_{env,0}$')
            axs[0,2].set_yticklabels([])



            axs[1,0].scatter(cr_array, ce0_array, color='k')
            axs[1,0].set_ylabel('$\chi_{env,0}$')

            axs[1,1].scatter(cl_array, ce0_array, color='k')
            axs[1,1].set_xlabel('$\chi_{lum}$')
            axs[1,1].set_yticklabels([])

            axs[1,2].axis('off')

            axs[2,0].scatter(cr_array, cl_array, color='k')
            axs[2,0].set_xlabel('$\chi_{red}$')
            axs[2,0].set_ylabel('$\chi_{lum}$')


            axs[2,1].axis('off')
            axs[2,2].axis('off')


            ellipse1_1sigma = Ellipse(xy=(mr.val, mi0.val), width=1*2*sr, height=1*2*si0, edgecolor='green', fc='None', lw=2)
            ellipse1_2sigma = Ellipse(xy=(mr.val, mi0.val), width=2*2*sr, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
            ellipse1_3sigma = Ellipse(xy=(mr.val, mi0.val), width=3*2*sr, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
            axs[0,0].add_patch(ellipse1_1sigma)
            axs[0,0].add_patch(ellipse1_2sigma)
            axs[0,0].add_patch(ellipse1_3sigma)


            ellipse2_1sigma = Ellipse(xy=(ml.val, mi0.val), width=1*2*sl, height=1*2*si0, edgecolor='green', fc='None', lw=2)
            ellipse2_2sigma = Ellipse(xy=(ml.val, mi0.val), width=2*2*sl, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
            ellipse2_3sigma = Ellipse(xy=(ml.val, mi0.val), width=3*2*sl, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
            axs[0,1].add_patch(ellipse2_1sigma)
            axs[0,1].add_patch(ellipse2_2sigma)
            axs[0,1].add_patch(ellipse2_3sigma)

            ellipse3_1sigma = Ellipse(xy=(me0.val, mi0.val), width=1*2*se0, height=1*2*si0, edgecolor='green', fc='None', lw=2)
            ellipse3_2sigma = Ellipse(xy=(me0.val, mi0.val), width=2*2*se0, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
            ellipse3_3sigma = Ellipse(xy=(me0.val, mi0.val), width=3*2*se0, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
            axs[0,2].add_patch(ellipse3_1sigma)
            axs[0,2].add_patch(ellipse3_2sigma)
            axs[0,2].add_patch(ellipse3_3sigma)



            ellipse4_1sigma = Ellipse(xy=(mr.val, me0.val), width=1*2*sr, height=1*2*se0, edgecolor='green', fc='None', lw=2)
            ellipse4_2sigma = Ellipse(xy=(mr.val, me0.val), width=2*2*sr, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
            ellipse4_3sigma = Ellipse(xy=(mr.val, me0.val), width=3*2*sr, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
            axs[1,0].add_patch(ellipse4_1sigma)
            axs[1,0].add_patch(ellipse4_2sigma)
            axs[1,0].add_patch(ellipse4_3sigma)


            ellipse5_1sigma = Ellipse(xy=(ml.val, me0.val), width=1*2*sl, height=1*2*se0, edgecolor='green', fc='None', lw=2)
            ellipse5_2sigma = Ellipse(xy=(ml.val, me0.val), width=2*2*sl, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
            ellipse5_3sigma = Ellipse(xy=(ml.val, me0.val), width=3*2*sl, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
            axs[1,1].add_patch(ellipse5_1sigma)
            axs[1,1].add_patch(ellipse5_2sigma)
            axs[1,1].add_patch(ellipse5_3sigma)



            ellipse6_1sigma = Ellipse(xy=(mr.val, ml.val), width=1*2*sr, height=1*2*sl, edgecolor='green', fc='None', lw=2)
            ellipse6_2sigma = Ellipse(xy=(mr.val, ml.val), width=2*2*sr, height=2*2*sl, edgecolor='cyan', fc='None', lw=2)
            ellipse6_3sigma = Ellipse(xy=(mr.val, ml.val), width=3*2*sr, height=3*2*sl, edgecolor='blue', fc='None', lw=2)
            axs[2,0].add_patch(ellipse6_1sigma)
            axs[2,0].add_patch(ellipse6_2sigma)
            axs[2,0].add_patch(ellipse6_3sigma)

            ellipse_prior1 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
            ellipse_prior2 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
            ellipse_prior3 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
            ellipse_prior4 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)
            ellipse_prior5 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)
            ellipse_prior6 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)

            axs[2,0].add_patch(ellipse_prior1)
            axs[1,0].add_patch(ellipse_prior2)
            axs[1,1].add_patch(ellipse_prior3)
            axs[0,2].add_patch(ellipse_prior4)
            axs[0,0].add_patch(ellipse_prior5)
            axs[0,1].add_patch(ellipse_prior6)

            plt.subplots_adjust(wspace=0, hspace=0)



            axs[0,0].axhline(y = 5.0, color = 'b', linestyle = '--') 
            axs[0,0].axvline(x = -0.5, color = 'b', linestyle='--')

            axs[1,0].axhline(y = 0.0, color = 'b', linestyle = '--') 
            axs[1,0].axvline(x = -0.5, color = 'b', linestyle='--')


            axs[2,0].axhline(y = 0.0, color = 'b', linestyle = '--') 
            axs[2,0].axvline(x = -0.5, color = 'b', linestyle='--')

            axs[0,1].axhline(y = 5.0, color = 'b', linestyle = '--') 
            axs[0,1].axvline(x = 0.0, color = 'b', linestyle='--')


            axs[1,1].axhline(y = 0.0, color = 'b', linestyle = '--') 
            axs[1,1].axvline(x = 0.0, color = 'b', linestyle='--')

            axs[0,2].axhline(y = 5.0, color = 'b', linestyle = '--') 
            axs[0,2].axvline(x = 0.0, color = 'b', linestyle='--')

            

        else:
            samples = ift.ResidualSampleList.load(self.results_path + 'pickle/last')
            c1=np.array([s for s in samples.iterator(self.ecomponents['chi1'])])
            m1, v1 = samples.sample_stat(self.ecomponents['chi1'])
            s1=np.sqrt(v1.val)
            print('chi1', m1.val, s1)




            c1_list=[]
            for i in range(0,len(c1)):
                c1_list.append(c1[i].val)
            c1_array=np.array(c1_list)

    
        
            plt.hist(c1_array, density=True, bins=10)
            plt.ylabel('#')
            plt.xlabel('$\chi_{1}$')



            plt.axvline(x = m1.val+s1, color = 'green', linestyle='--')
            plt.axvline(x = m1.val-s1, color = 'green', linestyle='--')

            plt.axvline(x = m1.val+2*s1, color = 'cyan', linestyle='--')
            plt.axvline(x = m1.val-2*s1, color = 'cyan', linestyle='--')

            plt.axvline(x = m1.val+3*s1, color = 'blue', linestyle='--')
            plt.axvline(x = m1.val-3*s1, color = 'blue', linestyle='--')


            plt.axvline(x = 4.0, color = 'black', linestyle='-')

            chi1= 4.0
            plt.axvline(x = chi1+0.5, color = 'red', linestyle='-')
            plt.axvline(x = chi1-0.5, color = 'red', linestyle='-')



        plt.savefig(f'{self.plot_path}EG_posterior.png', bbox_inches='tight')

        plt.show()
    

        


