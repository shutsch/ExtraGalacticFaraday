import libs as Egf 
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


class CatalogMaker():

    def __init__(self, params, base_catalog, dest_catalog=None):
        self.params = params
        self.base_catalog = base_catalog
        self.dest_catalog = dest_catalog


    def make_catalog(self):
        #seed
        np.random.seed(seed=self.params['params_inference.seed'])


        dest_data = self.dest_catalog if self.dest_catalog is not None else \
            self.base_catalog
        

        #eg and full data
        z_indices = Egf.get_data(self.base_catalog)['z indices']
        e_z = Egf.get_data(self.base_catalog)['e_z']
        e_F = Egf.get_data(self.base_catalog) ['e_F']
        rm_err = Egf.get_data(self.base_catalog) ['rm_err']
        
            
        #creation of mock F and z
        eF_samples=Egf.sampling_from_distribiution(self.params, e_z, e_F, self.base_catalog, dest_data)
        F_mock=eF_samples['F_mock']
        z_mock=eF_samples['z_mock']
        z_mock_indices=eF_samples['z_mock_indices']


        dest_data['z_best'][:] = np.nan
        dest_data['stokesI'][:] = np.nan
        dest_data['z_best'][z_mock_indices] = z_mock
        dest_data['stokesI'][z_mock_indices] = F_mock

        RM=Egf.rm(self.params, self.base_catalog, dest_data, rm_err, e_z, e_F, z_mock, F_mock, figname1='Mock_cat_plot_cat.png', figname2='Mock_cat_obs_vs_sim.png')

        noised_rm_data=RM['Noised data']
        sigma_mock=RM['Sigma noise mock']
                
     

        dest_data['rm'] = np.array(noised_rm_data.val)
        dest_data['rm_err'] =  sigma_mock
        

        catalog_name= Egf.write_to_file(self.params, dest_data)
        return catalog_name
        
