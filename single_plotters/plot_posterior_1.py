import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Posterior_Plotter_1():
    def __init__(self, samples, args):

        self.ecomponents = args['ecomponents']
        self.n_params= args['n_eg_params']
        self.results_path=args['results_path']
        self.plot_path=args['plot_path']
        self.samples = samples

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