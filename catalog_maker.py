from src.helper_functions.parameters_maker import Parameters_maker
import nifty8 as ift
import libs as Egf 
import numpy as np
import healpy as hp
from mock_seb23 import seb23
import matplotlib.pyplot as plt
import matplotlib
from nifty_cmaps import ncmap
matplotlib.use('TkAgg')
import utilities as U
import random
from astropy.cosmology import FlatLambdaCDM
import math as m

#cosmo and constants

#light_speed =  Egf.const['c']
#h =  Egf.const['Jens']['h']
#Wm = Egf.const['Jens']['Wm']
#Wc = Egf.const['Jens']['Wc']
#Wl = Egf.const['Jens']['Wl']
#H0 = 100 * h
    
#cosmo = FlatLambdaCDM(H0=H0, Om0=Wm)  
#L0 = float(Egf.const['L0'])
#D0 = Egf.const['D0']
#factor = float(Egf.const['factor'])



class CatalogMaker():

    def __init__(self, params, base_catalog, dest_catalog=None):
        self.params = params
        self.base_catalog = base_catalog
        self.dest_catalog = dest_catalog


    def make_catalog(self):
        #seed
        np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])

        #data catalogs
        data = self.base_catalog if self.base_catalog is not None else \
            Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

        dest_data = self.dest_catalog if self.dest_catalog is not None else \
            data
        
        #reading from base_catalog
        catalogs=Egf.get_data(data)

        #eg data
        z_indices = catalogs['z_indices']
        e_z = catalogs['e_z']
        e_F = catalogs['e_F']
        lerm = catalogs['lerm']         


        #full data
        rm_err = catalogs['rm_err']   



        sky_domain = ift.makeDomain(ift.HPSpace(self.params['params_inference.nside']))


        if self.params['params_mock_cat.maker_params.surveys.make_survey1'] == True:
            los=int(self.params['params_mock_cat.maker_params.multiple']*self.params['params_mock_cat.maker_params.surveys.los1'])
            b_sel_indices=np.where(abs(dest_data['b'][np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name1'])[0]])>self.params['params_mock_cat.maker_params.gal_lat_th'])[0] 
        else:
            los=int(self.params['params_mock_cat.maker_params.multiple']*dest_data['b'].size)
            b_sel_indices=np.where(abs(dest_data['b'])>self.params['params_mock_cat.maker_params.gal_lat_th'])[0] 


        #creation of indices of mock z
        z_mock_indices=np.unique(np.random.choice(b_sel_indices, size=los))
        lmock=len(z_mock_indices)
        print('Number of LOS with redshift', lmock)
        print('Total number of LOS in the catalog', dest_data['b'].size)


        #creation of mock F and z
        F_mock=Egf.sampling_from_distribiution(self.params, self.e_z, data, lmock)['F_mock']
        z_mock=Egf.sampling_from_distribiution(self.params, self.e_z, data, lmock) ['z_mock']




        dest_data['z_best'][:] = np.nan
        dest_data['stokesI'][:] = np.nan
        dest_data['z_best'][z_mock_indices] = z_mock
        dest_data['stokesI'][z_mock_indices] = F_mock


        # new filter
        dest_data_catalog = Egf.get_data(dest_data)

        #eg data
        z_indices = dest_data_catalog['z_indices'] 
        lerm = dest_data_catalog['lerm'] 

        eg_b = dest_data_catalog['b'] 

        theta_eg = dest_data_catalog['theta_eg'] 
        phi_eg = dest_data_catalog['phi_eg'] 


        ltheta=dest_data_catalog['ltheta'] 
        lthetaeg=dest_data_catalog['lthetaeg'] 

        eg_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(self.params['params_inference.nside'])), ift.makeDomain(ift.UnstructuredDomain(lthetaeg)), theta=theta_eg, phi=phi_eg)



        rm_gal=Egf.rm_gal(self.params, sky_domain)

        ### gal contribution in direction of eg points ####
        eg_gal_data = eg_projector(rm_gal)


        egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lerm,)))

        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
        egal_model_params = {'z': z_mock, 'F': F_mock, 'params': self.params}
        
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)


        rm_data=np.array(eg_gal_data.val)


        if self.params['params_mock_cat.maker_params.surveys.make_survey1'] == True:
          
            cat_index_1=np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name1'])[0]
            sigma_1 = rm_err[np.where(data['catalog']==self.params['params_mock_cat.maker_params.surveys.cat1'])[0]]
            sigma_mock=np.empty(dest_data['catalog'].size)
            sigma_mock[cat_index_1]=Egf.sampling_from_noise_distribiution(sigma_1, len(cat_index_1))
            print('sigma_mock cat1', sigma_mock[cat_index_1].mean())

            if self.params['params_mock_cat.maker_params.surveys.make_survey2'] == True:
                cat_index_2=np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name2'])[0]
                sigma_2 = rm_err[np.where(data['catalog']==self.params['params_mock_cat.maker_params.surveys.cat2'])[0]]
                sigma_mock[cat_index_2]=Egf.sampling_from_noise_distribiution(sigma_2, len(cat_index_2))
                print('sigma_mock cat2', sigma_mock[cat_index_2].mean())
                
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(dest_data['catalog'].size),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(sigma_mock_field.size), sampling_dtype=np.float64)

            rm_data+= N.draw_sample().val

        
            fig, axs = plt.subplots(2, 2)

            axs[0,0].hist(sigma_1, bins=100, density=True, color='green')
            
            axs[0,1].hist(sigma_2, bins=100, density=True, color='green')

            axs[1,0].hist(sigma_mock[cat_index_1], bins=100, density=True, color='lightgrey')
            axs[1,0].set_xlabel('$\\sigma_{1}$ (rad/m$^2$)')

            axs[1,1].hist(sigma_mock[cat_index_2], bins=100, density=True, color='lightgrey')
            axs[1,1].set_xlabel('$\\sigma_{2}$ (rad/m$^2$)')

            axs[1,1].sharex(axs[0,1])
            axs[1,0].sharex(axs[0,0])

            axs[0,0].set_xticks([])
            axs[0,1].set_xticks([])



            plt.subplots_adjust(wspace=0.5, hspace=0)
            plt.savefig('Noise.png', bbox_inches='tight')

        else:
            #creating mock sigma gal
            #NVSS cat 2009ApJ...702.1230T
            #LoTSS cat "LoTSS DR2 (O'Sullivan et al. 2022) "
            cat_index_gal=np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0][~np.isnan(np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0])]
            sigma_gal = data['rm_err'][cat_index_gal]
            #sigma_gal_mock=np.random.choice(sigma_gal,size=ltheta-lerm) 
            sigma_gal_mock=Egf.sampling_from_noise_distribiution(sigma_gal, ltheta-lerm)

            #sigma_gal_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(ltheta-lerm),np.array(sigma_gal_mock))
            #N_gal = ift.DiagonalOperator(sigma_gal_mock_field**2, domain=ift.UnstructuredDomain(ltheta-lerm), sampling_dtype=np.float64)
            #rm_data[np.isnan(data['z_best'])] +=  N_gal.draw_sample().val
            #print(rm_data.min(), rm_data.max(), rm_data.mean())

            #creating mock sigma eg
            cat_index_eg=np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0][~np.isnan(np.where(data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0])]
            sigma_eg = data['rm_err'][cat_index_eg]
            sigma_eg_mock=Egf.sampling_from_noise_distribiution(sigma_eg,lerm) 


            #sigma_eg_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(lerm),np.array(sigma_eg_mock))
            #N_eg = ift.DiagonalOperator(sigma_eg_mock_field**2, domain=ift.UnstructuredDomain(lerm), sampling_dtype=np.float64)
            #rm_data[z_indices]+= N_eg.draw_sample().val
            sigma_mock=np.empty(ltheta)
            sigma_mock[np.isnan(dest_data['z_best'])] = sigma_gal_mock
            sigma_mock[z_indices] = sigma_eg_mock
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(ltheta),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(ltheta), sampling_dtype=np.float64)

            rm_data+= N.draw_sample().val
            print('sigma_mock eg', sigma_eg_mock.mean())
            print('sigma_mock gal', sigma_gal_mock.mean())



            fig, axs = plt.subplots(2, 2)

            axs[0,0].hist(sigma_eg, bins=100, density=True, color='green')
            
            axs[0,1].hist(sigma_gal, bins=100, density=True, color='green')

            axs[1,0].hist(sigma_eg_mock, bins=100, density=True, color='lightgrey')
            axs[1,0].set_xlabel('$\\sigma_{eg}$ (rad/m$^2$)')

            axs[1,1].hist(sigma_gal_mock, bins=100, density=True, color='lightgrey')
            axs[1,1].set_xlabel('$\\sigma_{gal}$ (rad/m$^2$)')

            axs[1,1].sharex(axs[0,1])
            axs[1,0].sharex(axs[0,0])

            axs[0,0].set_xticks([])
            axs[0,1].set_xticks([])



            plt.subplots_adjust(wspace=0.5, hspace=0)
            plt.savefig('Noise.png', bbox_inches='tight')

        
        #modification of RM values to mimic wrong estimates present in the data and difficult to predict
        rm_data=Egf.npi(self.params, dest_data, rm_data, eg_b,eg_gal_data, sigma_mock)
        #adding eg contribution to the mock rm
        rm_data[z_indices]+=Egf.rm_eg(self.params, emodel, egal_mock_position, e_z, e_F, z_mock, F_mock) if self.params['params_mock_cat.maker_params.eg_on']==True else \
              0.0
        noised_rm_data=ift.makeField(ift.UnstructuredDomain(ltheta), rm_data)

                
        #Plot 1
        Egf.plot_mock(self.params,eg_projector.adjoint(eg_gal_data),eg_projector.adjoint(noised_rm_data),figname='Mock_cat_plot_cat.png')
        #Plot 2, questo plot non mi torna perche z_indices nel catalogo di partenza è diverso da quello del nuovo catalogo
        Egf.plot_mock_vs_observed(self.params, dest_data['rm'][z_indices],noised_rm_data.val[z_indices], dest_data['rm'][~z_indices],noised_rm_data.val[~z_indices], figname='Mock_cat_obs_vs_sim.png')


        dest_data['rm'] = np.array(noised_rm_data.val)
        dest_data['rm_err'] =  sigma_mock
        

        Egf.write_to_file(self.params, dest_data)
        
