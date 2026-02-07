import numpy as np
import libs as Egf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')




class Posterior_Plotter():
    def __init__(self, samples, args):

        self.samples = samples
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self, figname):
        #samples = ift.ResidualSampleList.load(f'{self.params["file_params.results_path"]}pickle/last')
        stats=Egf.sample_statistics(self, self.samples)

        width = 5.0
        points = 100

        fig, axs = plt.subplots(4, 4, figsize=(15, 15)) #, layout="constrained"

        plt.subplots_adjust(wspace=0, hspace=0)

        Egf.density_plot(self.params,axs,stats['chi_red_samples'],stats['chi_int_0_samples'],stats['chi_red_mean'], stats['chi_int_0_mean'],stats['chi_red_std'],stats['chi_int_0_std'], width, points, 0, 0, xlabel=None, ylabel='$\\chi_{int,0}$')
        Egf.density_plot(self.params,axs,stats['chi_lum_samples'],stats['chi_int_0_samples'],stats['chi_lum_mean'], stats['chi_int_0_mean'],stats['chi_lum_std'],stats['chi_int_0_std'], width, points, 0, 1)
        Egf.density_plot(self.params,axs,stats['chi_env_0_samples'],stats['chi_int_0_samples'],stats['chi_env_0_mean'], stats['chi_int_0_mean'],stats['chi_env_0_std'],stats['chi_int_0_std'], width, points, 0, 2, xlabel='$\\chi_{env,0}$')
        Egf.density_plot(self.params,axs,stats['chi_red_samples'],stats['chi_env_0_samples'],stats['chi_red_mean'],stats['chi_env_0_mean'],stats['chi_red_std'] ,stats['chi_env_0_std'], width, points, 1, 0, xlabel=None, ylabel='$\\chi_{env,0}$')
        Egf.density_plot(self.params,axs,stats['chi_lum_samples'],stats['chi_env_0_samples'],stats['chi_lum_mean'],stats['chi_env_0_mean'],stats['chi_lum_std'],stats['chi_env_0_std'], width, points, 1, 1, xlabel='$\\chi_{lum}$')
        Egf.density_plot(self.params,axs,stats['chi_red_samples'],stats['chi_lum_samples'],stats['chi_red_mean'],stats['chi_lum_mean'],stats['chi_red_std'],stats['chi_lum_std'], width, points, 2, 0, xlabel='$\\chi_{red}$', ylabel='$\\chi_{lum}$')
       

        Egf.histo_plot(self.params, axs,stats['chi_red_samples'], stats['chi_red_mean'], stats['chi_red_std'] , width, 3, 0, xlabel='$\\chi_{red}$')
        Egf.sigma_plot(self.params, axs, stats['chi_red_mean'], stats['chi_red_std'] , np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 3, 0)
        #amplitude parameter might need to be adjusted       
        Egf.gauss_plot(self.params, axs, stats['chi_red_mean'], stats['chi_red_std'] , width, 3, 0, 1000, label='Prior')

        Egf.histo_plot(self.params, axs,stats['chi_lum_samples'], stats['chi_lum_mean'], stats['chi_lum_std'], width, 2, 1, xlabel='$\\chi_{lum}$')
        Egf.sigma_plot(self.params, axs, stats['chi_lum_mean'], stats['chi_lum_std'], np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 2, 1)
        Egf.gauss_plot(self.params, axs, stats['chi_lum_mean'], stats['chi_lum_std'], width, 2, 1, 1000, label='Prior')

        Egf.histo_plot(self.params, axs,stats['chi_env_0_samples'], stats['chi_env_0_mean'], stats['chi_env_0_std'], width, 1, 2, xlabel='$\\chi_{env, 0}$')
        Egf.sigma_plot(self.params, axs, stats['chi_env_0_mean'], stats['chi_env_0_std'], np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 1, 2)
        Egf.gauss_plot(self.params, axs, stats['chi_env_0_mean'], stats['chi_env_0_std'], width, 1, 2, 1000, label='Prior')

        Egf.histo_plot(self.params, axs,stats['chi_int_0_samples'], stats['chi_int_0_mean'], stats['chi_int_0_std'], width, 0, 3, xlabel='$\\chi_{int, 0}$')
        Egf.sigma_plot(self.params, axs,  stats['chi_int_0_mean'], stats['chi_int_0_std'], np.array([1,2,3]), ['green', 'orange', 'red'], ['1-$\\sigma$', '2-$\\sigma$','3-$\\sigma$'], 0, 3)
        Egf.gauss_plot(self.params, axs,  stats['chi_int_0_mean'], stats['chi_int_0_std'], width, 0, 3, 1000, label='Prior')

        if self.params['params_inference.data_type']=='mock':
            axs[3,0].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_red'], color = 'k', linestyle = '-', label='Mock') 
            axs[2,1].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_lum'], color = 'k', linestyle = '-', label='Mock') 
            axs[1,2].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_env'], color = 'k', linestyle = '-', label='Mock') 
            axs[0,3].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_int'], color = 'k', linestyle = '-', label='Mock') 

            axs[0,0].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_int'], color = 'k', linestyle = '-') 
            axs[0,0].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_red'], color = 'k', linestyle='-')

            axs[1,0].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_env'], color = 'k', linestyle = '-') 
            axs[1,0].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_red'], color = 'k', linestyle='-')
            axs[2,0].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_lum'], color = 'k', linestyle = '-') 
            axs[2,0].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_red'], color = 'k', linestyle='-')
            axs[0,1].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_int'], color = 'k', linestyle = '-') 
            axs[0,1].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_lum'], color = 'k', linestyle='-')
            axs[1,1].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_env'], color = 'k', linestyle = '-') 
            axs[1,1].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_lum'], color = 'k', linestyle='-')
            axs[0,2].axhline(y = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_int'], color = 'k', linestyle = '-') 
            axs[0,2].axvline(x = self.params['params_mock_cat.mock_cat_initial_position.mean.mean_env'], color = 'k', linestyle='-')
            
        if self.params['params_inference.data_type']=='real':
            axs[3,0].axvline(x =  stats['chi_red_mean'], color = 'k', linestyle = '-', label='Mean') 
            axs[2,1].axvline(x =  stats['chi_lum_mean'], color = 'k', linestyle = '-', label='Mean') 
            axs[1,2].axvline(x =  stats['chi_env_0_mean'], color = 'k', linestyle = '-', label='Mean') 
            axs[0,3].axvline(x =  stats['chi_int_0_mean'], color = 'k', linestyle = '-', label='Mean') 

            axs[0,0].axhline(y = stats['chi_int_0_mean'], color = 'k', linestyle = '-') 
            axs[0,0].axvline(x = stats['chi_red_mean'], color = 'k', linestyle='-')

            axs[1,0].axhline(y = stats['chi_env_0_mean'], color = 'k', linestyle = '-') 
            axs[1,0].axvline(x = stats['chi_red_mean'], color = 'k', linestyle='-')
            axs[2,0].axhline(y = stats['chi_lum_mean'], color = 'k', linestyle = '-') 
            axs[2,0].axvline(x = stats['chi_red_mean'], color = 'k', linestyle='-')
            axs[0,1].axhline(y = stats['chi_int_0_mean'], color = 'k', linestyle = '-') 
            axs[0,1].axvline(x = stats['chi_lum_mean'], color = 'k', linestyle='-')
            axs[1,1].axhline(y = stats['chi_env_0_mean'], color = 'k', linestyle = '-') 
            axs[1,1].axvline(x = stats['chi_lum_mean'], color = 'k', linestyle='-')
            axs[0,2].axhline(y = stats['chi_lum_mean'], color = 'k', linestyle = '-') 
            axs[0,2].axvline(x = stats['chi_env_0_mean'], color = 'k', linestyle='-')

        axs[1,0].tick_params(labelbottom=True, direction='in')
        axs[0,1].tick_params(labelbottom=True, labelleft=False, direction='in')
        axs[0,2].tick_params(labelleft=False, direction='in')
        axs[1,1].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, direction='in')
        axs[0,0].tick_params(labelbottom=True, direction='in')
        axs[2,0].tick_params(labelbottom=True, labelleft=True, direction='in')

        axs[0,0].tick_params(axis='y', labelsize= self.params['plot.fontsize'])
        axs[1,0].tick_params(axis='y', labelsize= self.params['plot.fontsize'])
        axs[2,0].tick_params(axis='y', labelsize= self.params['plot.fontsize'])
        axs[3,0].tick_params(axis='x', labelsize= self.params['plot.fontsize'])
        axs[2,1].tick_params(axis='x', labelsize= self.params['plot.fontsize'])
        axs[1,2].tick_params(axis='x', labelsize= self.params['plot.fontsize'])
        axs[0,3].tick_params(axis='x', labelsize= self.params['plot.fontsize'])


        axs[1,0].sharex(axs[0,0])
        axs[2,0].sharex(axs[0,0])
        axs[0,1].sharex(axs[1,1])
        axs[0,1].sharey(axs[0,0])
        axs[0,2].sharey(axs[0,0])
        axs[1,1].sharey(axs[1,0])

        axs[2,2].axis('off')
        axs[3,1].axis('off')
        axs[3,2].axis('off')
        axs[3,3].axis('off')
        axs[2,3].axis('off')
        axs[1,3].axis('off')

        lines = []
        labels = []
        Line, Label = axs[0,3].get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)
        fig.legend(lines, labels, bbox_to_anchor=(0.05, 0.888), fontsize = self.params['plot.legend_fontsize'])

        plt.savefig(f'{self.params["file_params.plot_path"]}{figname}', bbox_inches='tight')
