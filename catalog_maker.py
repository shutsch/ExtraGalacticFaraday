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
import utilities as U
import random

class CatalogMaker():

    def __init__(self, params, base_catalog=None):
        self.params = params
        self.base_catalog = base_catalog
        #self.rng = np.random.default_rng(seed=params['params_mock_cat.maker_params.seed'])

    def make_catalog(self):
        sky_domain = ift.makeDomain(ift.HPSpace(self.params['params_inference.nside']))

        data = self.base_catalog if self.base_catalog is not None else \
            Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

        z_indices = ~np.isnan(data['z_best'])

        e_rm = np.array(data['rm'][z_indices])
        e_z = np.array(data['z_best'][z_indices])
        e_F = np.array(data['stokesI'][z_indices])

        los=self.params['params_mock_cat.maker_params.n_los']

        b45_indices=np.where(abs(data['b'])>45.0)[0]
        np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
        z_mock_indices=np.unique(np.random.choice(b45_indices, size=los))
        print('Number of LOS', len(z_mock_indices) )


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

            rm_gal, b, dm =seb23(self.params['params_mock_cat.maker_params.seed'])

            if self.params['params_mock_cat.maker_params.disk_on']==1:
                eg_gal_data = eg_projector(rm_gal)
            else:
                eg_gal_data = eg_projector(b)
            
            plot = ift.Plot()
            plot.add(dm, vmin=-250, vmax=250)
            plot.add(b, vmin=-2.50, vmax=2.50)
            plot.output(name='Mock_cat_Seb23_dm_b.png')

        else: #CONSISTENT catalog
            galactic_model = U.get_galactic_model(sky_domain, self.params)
            
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
        egal_model_params = {'z': e_z, 'F': e_F, 'params': self.params}
        
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)
        print(f'mock:{egal_mock_position.val}')

 

        rm_data=np.array(eg_gal_data.val)
        delta_rm_list=[]
        if self.params['params_mock_cat.maker_params.npi']==1:
            b_indices=np.where(np.isnan(data['z_best']))[0]
            np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
            npi_indices=np.unique(np.random.choice(b_indices, size=self.params['params_mock_cat.maker_params.npi_los']))
            print(npi_indices)
            print(npi_indices.size)
            print(ltheta-lerm)
            for item in b_indices:
                if item in npi_indices:
                    mu_nvss=self.params['params_mock_cat.maker_params.mu_nvss']
                    sigma_nvss=self.params['params_mock_cat.maker_params.sigma_nvss']
                    delta_rm=np.random.normal(mu_nvss, sigma_nvss)
                    if random.choice('+-')=='-':
                        rm_data[item] -= delta_rm
                        delta_rm_list.append(-delta_rm)
                    else:
                        rm_data[item] += delta_rm
                        delta_rm_list.append(delta_rm)
                
            delta_rm_array=np.array(delta_rm_list)
            plt.scatter(eg_b[npi_indices], delta_rm_array)
            plt.savefig('Delta_rm.png', bbox_inches='tight')

        

        #creating mock sigma gal
        #NVSS cat 2009ApJ...702.1230T
        #LoTSS cat "LoTSS DR2 (O'Sullivan et al. 2022)"
        cat_index_gal=np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0][~np.isnan(np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0])]
        sigma_gal = data['rm_err'][cat_index_gal]
        histogram_sigma_gal = rv_histogram(np.histogram(sigma_gal, bins=10000), density=False)
        sigma_gal_mock=histogram_sigma_gal.rvs(size=ltheta-lerm)
        sigma_gal_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(ltheta-lerm),np.array(sigma_gal_mock))
        N_gal = ift.DiagonalOperator(sigma_gal_mock_field**2, domain=ift.UnstructuredDomain(ltheta-lerm), sampling_dtype=np.float64)


        ### rm data assembly ###
        rm_data[np.isnan(data['z_best'])] +=  N_gal.draw_sample().val
        print(rm_data.min(), rm_data.max(), rm_data.mean())


        if self.params['params_mock_cat.maker_params.eg_on']==1:
            #creating mock sigma eg
            cat_index_eg=np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0][~np.isnan(np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0])]
            sigma_eg = data['rm_err'][cat_index_eg]
            histogram_sigma_eg = rv_histogram(np.histogram(sigma_eg, bins=100), density=False)
            sigma_eg_mock=histogram_sigma_eg.rvs(size=lerm)
            sigma_eg_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(lerm),np.array(sigma_eg_mock))
            N_eg = ift.DiagonalOperator(sigma_eg_mock_field**2, domain=ift.UnstructuredDomain(lerm), sampling_dtype=np.float64)

            np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
            rand_rm=np.random.normal(0.0, 1.0,len(e_rm))
            egal_contr = emodel.get_model().sqrt()(egal_mock_position).val*rand_rm
            rm_data[z_indices]+=egal_contr + N_eg.draw_sample().val
            print('std',np.std(egal_contr))
            print('mean',np.mean(egal_contr))

        noised_rm_data=ift.makeField(ift.UnstructuredDomain(ltheta), rm_data)
        #Print noise
        print('Gal noise', N_gal.draw_sample().val.std())
        print('Egal noise', N_eg.draw_sample().val.std())

        #Plot 1
        plot = ift.Plot()
        plot.add(eg_projector.adjoint(eg_gal_data), vmin=-250, vmax=250)
        plot.add(eg_projector.adjoint(noised_rm_data), vmin=-250, vmax=250)
        plot.output(name='Mock_cat_plot_cat.png')
        #plt.savefig('Mock_cat_plot_cat.png', bbox_inches='tight')

        #Plot 2
        fig, axs = plt.subplots(1, 2)

        axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
        axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
        axs[1].scatter(data['rm'][z_indices],noised_rm_data.val[z_indices])
        axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
        axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
        axs[0].scatter(data['rm'][~z_indices],noised_rm_data.val[~z_indices])
        # plt.show()
        plt.savefig('Mock_cat_obs_vs_sim.png', bbox_inches='tight')

        data['rm'] = np.array(noised_rm_data.val)
        data['rm_err'][np.isnan(data['z_best'])] =  sigma_gal_mock*np.ones(ltheta-lerm)
        data['rm_err'][~np.isnan(data['z_best'])] =  sigma_eg_mock*np.ones(lerm)
        
        hdu= fits.open(self.params['params_inference.cat_path']+'master_catalog_vercustom.fits')
        hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = data['rm']
        hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  data['rm_err']
        hdu[1].data['z_best'][np.where(hdu[1].data['type']!='Pulsar')] =  data['z_best']
        hdu[1].data['stokesI'][np.where(hdu[1].data['type']!='Pulsar')] =  data['stokesI']
        hdu.writeto(self.params['params_inference.cat_path']+'master_catalog_vercustom_sim.fits', overwrite=True)
        hdu.close()

