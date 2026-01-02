import numpy as np
import nifty8 as ift
import libs as Egf 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')





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

        
        
        #mean,var=samples.sample_stat()
        #sl = samples.at(mean)
        
        egal_var=np.array([emodel.get_model().force(s).val for s in samples.iterator()])

        rand_rm=np.random.normal(0.0, 1.0, egal_var.shape[1])
        egal_contr = np.sqrt(egal_var)*rand_rm
        eg_std=np.std(egal_contr.flatten())
        eg_mean=np.mean(egal_contr.flatten())
        print(egal_contr.shape)
        print('eg_std', eg_std)
        print('eg_mean', eg_mean)
       
        fig, ax = plt.subplots()
        ax.hist(egal_contr.flatten(), bins=1000, color='skyblue', edgecolor='black', range=(-200.0,200.0), density=True)
        Egf.draw_text(ax,eg_mean,eg_std)
        ax.set_title('Total')
        plt.xlabel('$\\phi_{eg}$ [rad m$^{-2}$]')
        plt.ylabel('Occurrency')
        plt.savefig(f'{self.params["file_params.plot_path"]}{figname_distribution}', bbox_inches='tight')
        plt.clf()

        

