import numpy as np
import nifty8 as ift
import libs as Egf 
import matplotlib.pyplot as plt
import matplotlib
import math as m
import healpy as hp
from matplotlib import cm
from astropy.cosmology import FlatLambdaCDM
from src.helper_functions.misc import gal2gal
matplotlib.use('TkAgg')

#cosmo and constants

light_speed =  Egf.const['c']
h =  Egf.const['Jens']['h']
Wm = Egf.const['Jens']['Wm']
Wc = Egf.const['Jens']['Wc']
Wl = Egf.const['Jens']['Wl']
H0 = 100 * h
    
cosmo = FlatLambdaCDM(H0=H0, Om0=Wm)  
L0 = float(Egf.const['L0'])
D0 = Egf.const['D0']
factor = float(Egf.const['factor'])



class Map_Plotter():
    def __init__(self, args):
        self.emodel=args['emodel']
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self, figname_distribution):
        params= self.params
        emodel=self.emodel
        samples = ift.ResidualSampleList.load(f'{self.params["file_params.results_path"]}pickle/last')

        mr, vr = samples.sample_stat(self.ecomponents['chi_red'])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])
        ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        
        sr=np.sqrt(vr.val)
        si0=np.sqrt(vi0.val)
        sl=np.sqrt(vl.val)
        se0=np.sqrt(ve0.val)

        print('cr', mr.val, 'pm', sr)
        print('ci0', mi0.val, 'pm', si0)
        print('cl', ml.val, 'pm', sl)
        print('ce0', me0.val, 'pm', se0)

        
        
        mean,var=samples.sample_stat()
        sl = samples.at(mean)
        egal_var=np.array([emodel.get_model().force(s).val for s in sl.iterator()])

        np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
        rand_rm=np.random.normal(0.0, 1.0,len(e_z))
        egal_contr = np.sqrt(egal_var)*rand_rm
        print('eg_std', np.std(egal_contr))

        plt.hist(egal_contr.flatten(), bins=1000, color='skyblue', edgecolor='black')
        plt.xlabel('$\\phi_{eg}$ [rad m$^{-2}$]')
        plt.ylabel('Occurrency')
        plt.savefig(f'{self.params["file_params.plot_path"]}{figname_distribution}', bbox_inches='tight')
        plt.clf()

