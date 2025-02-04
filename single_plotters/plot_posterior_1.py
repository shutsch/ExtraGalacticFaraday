import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
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

        fig, ax = plt.subplots()
        ax.hist(c1_array, density=True, bins=10)
        ax.set_ylabel('#')
        ax.set_xlabel('$\chi_{1}$')

        ax.axvline(x = m1.val+s1, color = 'green', linestyle='--')
        ax.axvline(x = m1.val-s1, color = 'green', linestyle='--')

        ax.axvline(x = m1.val+2*s1, color = 'cyan', linestyle='--')
        ax.axvline(x = m1.val-2*s1, color = 'cyan', linestyle='--')

        ax.axvline(x = m1.val+3*s1, color = 'blue', linestyle='--')
        ax.axvline(x = m1.val-3*s1, color = 'blue', linestyle='--')

        ax.axvline(x = 4.0, color = 'black', linestyle='-')

        
        ax.axvline(x = self.params['prior_mean.prior_mean_one']+self.params['prior_std.prior_std_one'], color = 'red', linestyle='-')
        ax.axvline(x = self.params['prior_mean.prior_mean_one']-self.params['prior_std.prior_std_one'], color = 'red', linestyle='-')

        ax.set_xlim(m1.val-5*s1,m1.val+5*s1)
        
        ax.tick_params(axis='y',direction='in')
        ax.tick_params(axis='x',direction='in')