import nifty8 as ift
import libs as Egf 
import numpy as np
from src.helper_functions.misc import gal2gal
from mock_seb23 import seb23
from astropy.io import fits
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
import matplotlib
from src.helper_functions.parameters_maker import Parameters_maker
matplotlib.use('TkAgg')
import sys

class CatalogMaker():

    def __init__(self, params):
        self.params = params
        ift.random.push_sseq_from_seed(params['params_mock_cat.maker_params.seed_cat'])

    def make_catalog(self):

        sky_domain = ift.makeDomain(ift.HPSpace(self.params['params.nside']))

        #starting catalog
        data = Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

        z_indices = ~np.isnan(data['z_best'])

        e_rm = np.array(data['rm'][z_indices])
        e_z = np.array(data['z_best'][z_indices])
        e_F = np.array(data['stokesI'][z_indices])

        los=self.params['params.n_los']

        b_indices=np.where(abs(data['b'])>45.0)[0]
        z_mock_indices=np.unique(np.random.choice(b_indices, size=los))

        histogram_z = rv_histogram(np.histogram(e_z, bins=100), density=False)
        z_mock=histogram_z.rvs(size=z_mock_indices.size)

        data['z_best'][:] = np.nan
        data['z_best'][z_mock_indices] = z_mock

        F_all = np.array(data['stokesI'])
        F_indices = np.where(F_all>0)[0]
        F_sample= np.array(data['stokesI'][F_indices])

        #creating mock fluxes
        histogram_F = rv_histogram(np.histogram(F_sample, bins=10000), density=False)
        F_mock=histogram_F.rvs(size=len(data['stokesI']))

        data['stokesI'] = F_mock

        # new filter
        z_indices = ~np.isnan(data['z_best'])
        e_z = np.array(data['z_best'][z_indices])
        e_F = np.array(data['stokesI'][z_indices])
        e_rm = np.array(data['rm'][z_indices])
        #g_rm = np.array(data['rm'][~z_indices])
        lerm = len(e_rm)

        eg_l = np.array(data['l'])
        eg_b = np.array(data['b'])

        theta_eg, phi_eg = gal2gal(eg_l, eg_b) # converting to colatitude and logitude in radians

        ltheta=len(data['theta'])
        lthetaeg = len(theta_eg)
        
        eg_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(256)), ift.makeDomain(ift.UnstructuredDomain(lthetaeg)), theta=theta_eg, phi=phi_eg)

        if(self.params['params_mock_cat.maker_params.maker_type'] == "seb23"):

            rm_gal, b, dm =seb23(self.params['params_mock_cat.maker_params.seed_cat'])

            if self.params['params_mock_cat.maker_params.disk_on']==1:
                eg_gal_data = eg_projector(rm_gal)
            else:
                eg_gal_data = eg_projector(b)
            
            plot = ift.Plot()
            plot.add(dm, vmin=-250, vmax=250)
            plot.add(b, vmin=-2.50, vmax=2.50)
            plot.output()
            plt.savefig('Mock_cat_Seb23_dm_b.png', bbox_inches='tight')

        else: #CONSISTENT catalog
    
            log_amplitude_params = {'fluctuations': {'asperity': self.params['params_mock_cat.log_amplitude.fluctuations.asperity'], 
                                                'flexibility': self.params['params_mock_cat.log_amplitude.fluctuations.flexibility'],  
                                                'fluctuations': self.params['params_mock_cat.log_amplitude.fluctuations.fluctuations'], 
                                                'loglogavgslope': self.params['params_mock_cat.log_amplitude.fluctuations.loglogavgslope'], },
                                'offset': {'offset_mean': self.params['params_mock_cat.log_amplitude.offset.offset_mean'], 
                                        'offset_std': self.params['params_mock_cat.log_amplitude.offset.offset_std']},}

            sign_params = {'fluctuations': {'asperity': self.params['params_mock_cat.sign.fluctuations.asperity'], 
                                                'flexibility': self.params['params_mock_cat.sign.fluctuations.flexibility'],  
                                                'fluctuations': self.params['params_mock_cat.sign.fluctuations.fluctuations'], 
                                                'loglogavgslope': self.params['params_mock_cat.sign.fluctuations.loglogavgslope'], },
                                'offset': {'offset_mean': self.params['params_mock_cat.sign.offset.offset_mean'], 
                                        'offset_std': self.params['params_mock_cat.sign.offset.offset_std']},}

            galactic_model = Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                            'sign_parameters': sign_params})
            
            gal_mock_position = ift.from_random(galactic_model.get_model().domain, 'normal')
            gal=galactic_model.get_model()(gal_mock_position)

            plot = ift.Plot()
            plot.add(gal, vmin=-250, vmax=250)
            plot.output()
            plt.savefig('Mock_cat_consistent_RM_gal.png', bbox_inches='tight')

            ### eg contribution ####
            eg_gal_data = eg_projector(gal)

        egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lerm,)))

        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
        egal_model_params = {'z': e_z, 'F': e_F, 'n_params': self.params['params.n_eg_params']  }
        
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

        #egal_mock_position = ift.from_random(emodel.get_model().domain, 'normal')
        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)
        print(f'mock:{egal_mock_position.val}')

        ### Specify noise
        noise = 0.05
        N = ift.ScalingOperator(ift.UnstructuredDomain(ltheta), noise, np.float64)

        ### rm data assembly ###
        rm_data=np.array(eg_gal_data.val)
        print(rm_data.min(), rm_data.max(), rm_data.mean())
        rand_rm=np.random.normal(0.0, 1.0,len(e_rm))
        egal_contr = emodel.get_model().sqrt()(egal_mock_position).val*rand_rm

        rm_data[z_indices]+=egal_contr
        print('std',np.std(egal_contr))
        print('mean',np.mean(egal_contr))

        rm_data_field=ift.makeField(ift.UnstructuredDomain(ltheta), rm_data)

        noised_rm_data = rm_data_field + N.draw_sample()

        #Plot 1
        plot = ift.Plot()
        plot.add(eg_projector.adjoint(eg_gal_data), vmin=-250, vmax=250)
        plot.add(eg_projector.adjoint(noised_rm_data), vmin=-250, vmax=250)
        plot.output()
        plt.savefig('Mock_cat_plot_cat.png', bbox_inches='tight')

        #Plot 2
        fig, axs = plt.subplots(1, 2)

        axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
        axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
        axs[1].scatter(data['rm'][z_indices],noised_rm_data.val[z_indices])
        axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
        axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
        axs[0].scatter(data['rm'][~z_indices],noised_rm_data.val[~z_indices])
        plt.show()
        plt.savefig('Mock_cat_obs_vs_sim.png', bbox_inches='tight')

        data['rm'] = np.array(noised_rm_data.val)
        data['rm_err'] =  noise*np.ones(np.array(noised_rm_data.val).size)
        
        hdu= fits.open(self.params['params.cat_path']+'master_catalog_vercustom.fits')
        hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = data['rm']
        hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  data['rm_err']
        hdu.writeto(self.params['params.cat_path']+'master_catalog_vercustom_sim.fits', overwrite=True)
        hdu.close()

