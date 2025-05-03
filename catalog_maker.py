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

class CatalogMaker():

    def __init__(self, params, src_catalog=None, dest_catalog=None):
        self.params = params
        self.src_catalog = src_catalog
        self.dest_catalog = dest_catalog

        self.src_data = self.src_catalog if self.src_catalog is not None else \
            Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

        self.dest_data = self.dest_catalog if self.dest_catalog is not None else \
            self.src_catalog     

        

    def make_catalog(self):
        src_data=self.src_data
        dest_data=self.dest_data

        sky_domain = ift.makeDomain(ift.HPSpace(self.params['params_inference.nside']))

        z_indices = ~np.isnan(src_data['z_best'])

        e_rm = np.array(src_data['rm'][z_indices])
        e_z = np.array(src_data['z_best'][z_indices])
        e_F = np.array(src_data['stokesI'][z_indices])

        los=self.params['params_mock_cat.maker_params.n_los']

        if self.params['params_mock_cat.maker_params.polar_caps'] == True:
            if self.params['params_mock_cat.maker_params.surveys.make_survey'] == True:
                b_sel_indices=np.where(abs(dest_data['b'][np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name'])[0]])>self.params['params_mock_cat.maker_params.elev_th'])[0] 
            else:
                b_sel_indices=np.where(abs(dest_data['b'])>self.params['params_mock_cat.maker_params.elev_th'])[0] 
        if self.params['params_mock_cat.maker_params.make_fraction'] == True:
            if self.params['params_mock_cat.maker_params.surveys.make_survey'] == True:
                los=int(self.params['params_mock_cat.maker_params.fraction']*self.params['params_mock_cat.maker_params.surveys.los'])
                b_sel_indices=np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name'])[0] 
            else:
                los=int(self.params['params_mock_cat.maker_params.fraction']*dest_data['b'].size)
                b_sel_indices=np.arange(0,dest_data['b'].size) 
        
        np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
        z_mock_indices=np.unique(np.random.choice(b_sel_indices, size=los))
        print('Number of LOS with redshift', len(z_mock_indices) )


        histogram_z = rv_histogram(np.histogram(e_z, bins=100), density=False)
        z_mock=histogram_z.rvs(size=z_mock_indices.size)

        dest_data['z_best'][:] = np.nan
        dest_data['z_best'][z_mock_indices] = z_mock


        F_all = np.array(src_data['stokesI'])
        F_indices = np.where(F_all>0)[0]
        F_sample= np.array(src_data['stokesI'][F_indices])

        #creating mock fluxes
        histogram_F = rv_histogram(np.histogram(F_sample, bins=10000), density=False)
        F_mock=histogram_F.rvs(size=len(dest_data['stokesI']))

        dest_data['stokesI'] = F_mock

        # new filter
        z_indices = ~np.isnan(dest_data['z_best'])
        e_z = np.array(dest_data['z_best'][z_indices])
        e_F = np.array(dest_data['stokesI'][z_indices])
        e_rm = np.array(dest_data['rm'][z_indices])
        lerm = len(e_rm)

        eg_l = np.array(dest_data['l'])
        eg_b = np.array(dest_data['b'])

        theta_eg, phi_eg = gal2gal(eg_l, eg_b) # converting to colatitude and logitude in radians

        ltheta=len(dest_data['theta'])
        lthetaeg = len(theta_eg)
        
        eg_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(256)), ift.makeDomain(ift.UnstructuredDomain(lthetaeg)), theta=theta_eg, phi=phi_eg)

        if(self.params['params_mock_cat.maker_params.maker_type'] == "seb23"):

            rm_gal, b, dm =seb23(self.params)

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
            plot.saoutput(name='Mock_cat_consistent_RM_gal.png')

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


        if self.params['params_mock_cat.maker_params.surveys.make_survey'] == True:
          
            cat_index_1=np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name'])[0]
            sigma_1 = src_data['rm_err'][np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.surveys.cat'])[0]]
            histogram_sigma_1 = rv_histogram(np.histogram(sigma_1, bins=100), density=False)
            sigma_1_mock=histogram_sigma_1.rvs(size=cat_index_1.size)
            sigma_mock=np.empty(dest_data['catalog'].size)
            sigma_mock[cat_index_1]=sigma_1_mock
            print('sigma_mock cat1', sigma_1_mock.mean())

            if self.params['params_mock_cat.maker_params.surveys.make_survey2'] == True:
                cat_index_2=np.where(dest_data['catalog']==self.params['params_mock_cat.maker_params.surveys.name2'])[0]
                sigma_2 = src_data['rm_err'][np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.surveys.cat2'])[0]]
                histogram_sigma_2 = rv_histogram(np.histogram(sigma_2, bins=100), density=False)
                sigma_2_mock=histogram_sigma_2.rvs(size=cat_index_2.size)
                sigma_mock[cat_index_2]=sigma_2_mock
                print('sigma_mock cat2', sigma_2_mock.mean())
                
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(dest_data['catalog'].size),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(sigma_mock_field.size), sampling_dtype=np.float64)

            rm_data+= N.draw_sample().val
            
            

        else:

            #creating mock sigma gal
            #NVSS cat 2009ApJ...702.1230T
            #LoTSS cat "LoTSS DR2 (O'Sullivan et al. 2022) "
            cat_index_gal=np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0][~np.isnan(np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.cat_gal'])[0])]
            sigma_gal = src_data['rm_err'][cat_index_gal]
            histogram_sigma_gal = rv_histogram(np.histogram(sigma_gal, bins=10000), density=False)
            sigma_gal_mock=histogram_sigma_gal.rvs(size=ltheta-lerm)


            #creating mock sigma eg
            cat_index_eg=np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0][~np.isnan(np.where(src_data['catalog']==self.params['params_mock_cat.maker_params.cat_eg'])[0])]
            sigma_eg = src_data['rm_err'][cat_index_eg]
            histogram_sigma_eg = rv_histogram(np.histogram(sigma_eg, bins=100), density=False)
            sigma_eg_mock=histogram_sigma_eg.rvs(size=lerm)



            sigma_mock=np.empty(ltheta)
            sigma_mock[np.isnan(dest_data['z_best'])] = sigma_gal_mock
            sigma_mock[z_indices] = sigma_eg_mock
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(ltheta),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(ltheta), sampling_dtype=np.float64)

            rm_data+= N.draw_sample().val
            print('sigma_mock eg', sigma_eg_mock.mean())
            print('sigma_mock gal', sigma_gal_mock.mean())

        if self.params['params_mock_cat.maker_params.eg_on']==True:
            np.random.seed(seed=self.params['params_mock_cat.maker_params.seed'])
            rand_rm=np.random.normal(0.0, 1.0,len(e_rm))
            #rand_rm=self.rng.normal(0.0, 1.0,len(e_rm))
            egal_contr = emodel.get_model().sqrt()(egal_mock_position).val*rand_rm
            rm_data[z_indices]+=egal_contr 
            print('std',np.std(egal_contr))
            print('mean',np.mean(egal_contr))

        noised_rm_data=ift.makeField(ift.UnstructuredDomain(ltheta), rm_data)




        #Plot 1
        plot = ift.Plot()
        plot.add(eg_projector.adjoint(eg_gal_data), vmin=-250, vmax=250)
        plot.add(eg_projector.adjoint(noised_rm_data), vmin=-250, vmax=250)
        plot.output(name='Mock_cat_plot_cat.png')

        #Plot 2
        fig, axs = plt.subplots(1, 2)

        axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
        axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
        axs[1].scatter(dest_data['rm'][z_indices],noised_rm_data.val[z_indices])
        axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
        axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
        axs[0].scatter(dest_data['rm'][~z_indices],noised_rm_data.val[~z_indices])
        # plt.show()
        plt.savefig('Mock_cat_obs_vs_sim.png', bbox_inches='tight')

        dest_data['rm'] = np.array(noised_rm_data.val)
        dest_data['rm_err'] =  sigma_mock
        #dest_data['rm_err'][np.isnan(dest_data['z_best'])] =  sigma_gal_mock*np.ones(ltheta-lerm)
        #dest_data['rm_err'][~np.isnan(dest_data['z_best'])] =  sigma_eg_mock*np.ones(lerm)
        
        if self.params['params_mock_cat.maker_params.surveys.make_survey']==True:
            hdu= fits.open(self.params['params_inference.cat_path']+ self.params['params_mock_cat.maker_params.surveys.name']+'_catalog.fits')
        else:
           hdu= fits.open(self.params['params_inference.cat_path']+'master_catalog_vercustom.fits')
        hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = dest_data['rm']
        hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['rm_err']
        hdu[1].data['z_best'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['z_best']
        hdu[1].data['stokesI'][np.where(hdu[1].data['type']!='Pulsar')] =  dest_data['stokesI']
        hdu.writeto(self.params['params_inference.cat_path']+'master_catalog_vercustom_sim.fits', overwrite=True)
        hdu.close()

