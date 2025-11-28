import nifty8 as ift
import libs as Egf 
import numpy as np



def rm(params, data, dest_data, rm_err, e_z, e_F, z_mock, F_mock, figname1, figname2):


        sky_domain = ift.makeDomain(ift.HPSpace(params['params_inference.nside']))


        # new filter
        dest_data_catalog = Egf.get_data(dest_data)

        #eg data
        z_indices = dest_data_catalog['z indices'] 
        lerm = dest_data_catalog['lerm'] 

        eg_b = dest_data_catalog['eg_b'] 

        theta_eg = dest_data_catalog['theta_eg'] 
        phi_eg = dest_data_catalog['phi_eg'] 


        ltheta=dest_data_catalog['ltheta'] 
        lthetaeg=dest_data_catalog['lthetaeg'] 

        eg_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(params['params_inference.nside'])), ift.makeDomain(ift.UnstructuredDomain(lthetaeg)), theta=theta_eg, phi=phi_eg)



        rm_gal=Egf.rm_gal(params, sky_domain)

        ### gal contribution in direction of eg points ####
        eg_gal_data = eg_projector(rm_gal)


        egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lerm,)))

        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
        egal_model_params = {'z': z_mock, 'F': F_mock, 'params': params}
        
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)


        rm_data=np.array(eg_gal_data.val)

        # QUI UNA FUNZIONE PER IL NOISE
        rm_noise=Egf.rm_noise(params, data, dest_data, rm_err, ltheta, lerm, z_indices, 'Noise.png')
        noise=rm_noise['Noise']
        sigma_mock =rm_noise['Sigma noise mock']
        rm_data+=noise


        #modification of RM values to mimic wrong estimates present in the data and difficult to predict
        rm_data=Egf.npi(params, dest_data, rm_data, eg_b,eg_gal_data, sigma_mock)
        #adding eg contribution to the mock rm
        rm_data[z_indices]+=Egf.rm_eg(params, emodel, egal_mock_position, e_z, e_F, z_mock, F_mock) if params['params_mock_cat.maker_params.eg_on']==True else \
              0.0
        noised_rm_data=ift.makeField(ift.UnstructuredDomain(ltheta), rm_data)

        #Plot 1
        Egf.plot_mock(params,eg_projector.adjoint(eg_gal_data),eg_projector.adjoint(noised_rm_data),figname1)
        #Plot 2, questo plot non mi torna perche z_indices nel catalogo di partenza è diverso da quello del nuovo catalogo
        Egf.plot_mock_vs_observed(params, dest_data['rm'][z_indices],noised_rm_data.val[z_indices], dest_data['rm'][~z_indices],noised_rm_data.val[~z_indices], figname2)


        return {'Noised data': noised_rm_data, 'Sigma noise mock': sigma_mock}