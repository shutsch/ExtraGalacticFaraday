import nifty8 as ift
import libs as Egf 
import numpy as np
from src.helper_functions.misc import gal2gal
from mock_seb23 import seb23
from astropy.io import fits
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def run_inference():
    


    # set the HealPix resolution parameter and the sky domain

    sky_domain = ift.makeDomain(ift.HPSpace(Egf.config['params']['nside']))

    # load_the data, define domains, covariance and projection operators

    data = Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

    # filter
    z_indices = ~np.isnan(data['z_best'])

    e_rm = np.array(data['rm'][z_indices])
    #e_stddev = np.array(data['rm_err'][z_indices])
    e_z = np.array(data['z_best'][z_indices])
    e_F = np.array(data['stokesI'][z_indices])


    los=14500

    b_indices=np.where(abs(data['b'])>45.0)[0]
    z_mock_indices=np.unique(np.random.choice(b_indices, size=los))

    #creating mock redshifts
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
    g_rm = np.array(data['rm'][~z_indices])


    ### gal contribution ####

    rm_gal, b, dm =seb23(seed)


    o_l = np.array(data['l'])
    o_b = np.array(data['b'])

    theta_o, phi_o = gal2gal(o_l, o_b) # converting to colatitude and logitude in radians

    o_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(256)), ift.makeDomain(ift.UnstructuredDomain(len(theta_o))), theta=theta_o, phi=phi_o)

    o_rm_gal_data = o_projector(rm_gal)
    #o_rm_gal_data = o_projector(b) #whitouth disk



    ### eg contribution ####

    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(e_rm),)))


    # build the full model and connect it to the likelihood
    # set the extra-galactic model hyper-parameters and initialize the model
    egal_model_params = {'z': e_z, 'F': e_F }
      
    emodel = Egf.ExtraGalDemoModel(egal_data_domain, egal_model_params)
 


    egal_mock_position = ift.from_random(emodel.get_model().domain, 'normal')

    p_d = egal_mock_position.to_dict() 

    #p_d['chi_int_0'] = ift.full(p_d['chi_int_0'].domain, 0.0)
    #p_d['chi_env_0'] = ift.full(p_d['chi_env_0'].domain, 0.0)
    #p_d['chi_lum'] = ift.full(p_d['chi_lum'].domain, 0.0)
    #p_d['chi_red'] = ift.full(p_d['chi_red'].domain, 0.0)

    p_d['chi1'] = ift.full(p_d['chi1'].domain, 0.0)

    egal_mock_position = egal_mock_position.from_dict(p_d)



    ### Specify noise
    noise = 0.05
    N = ift.ScalingOperator(ift.UnstructuredDomain(len(data['theta'])), noise, np.float64)

    ### rm data assembly ###

    rm_data=np.array(o_rm_gal_data.val)
    print(rm_data.min(), rm_data.max(), rm_data.mean())
    rm_data[z_indices]+=emodel.get_model().sqrt()(egal_mock_position).val*np.random.normal(0.0, 1.0,len(e_rm))
    print('std',np.std(emodel.get_model().sqrt()(egal_mock_position).val*np.random.normal(0.0, 1.0,len(e_rm))))
    print('mean',np.mean(emodel.get_model().sqrt()(egal_mock_position).val*np.random.normal(0.0, 1.0,len(e_rm))))

    rm_data_field=ift.makeField(ift.UnstructuredDomain(len(data['theta'])), rm_data)

    noised_rm_data = rm_data_field + N.draw_sample()

    #sign = components['sign'].force(mock_position)

    #Plot 1
    #plot = ift.Plot()
    #plot.add(o_projector.adjoint(o_rm_gal_data), vmin=-0.25, vmax=0.25)
    #plot.add(o_projector.adjoint(noised_rm_data), vmin=-0.25, vmax=0.25)
    #plot.add(b, vmin=-10, vmax=10)
    #plot.output()

    #Plot 2
    fig, axs = plt.subplots(1, 2)

    axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
    axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
    axs[1].scatter(data['rm'][z_indices],noised_rm_data.val[z_indices])
    axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
    axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
    axs[0].scatter(data['rm'][~z_indices],noised_rm_data.val[~z_indices])
    plt.show()




    data['rm'] = np.array(noised_rm_data.val)
    data['rm_err'] =  noise*np.ones(np.array(noised_rm_data.val).size)
    
    hdu= fits.open('/home/valentina/Documents/PROJECTS/BAYESIAN_CODE/DEFROST/ExtraGalacticFaraday/data/Faraday/catalog_versions/master_catalog_vercustom.fits')
    hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = np.array(noised_rm_data.val)
    hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  noise*np.ones(np.array(noised_rm_data.val).size)
    #hdu.writeto('/home/valentina/Documents/PROJECTS/BAYESIAN_CODE/DEFROST/ExtraGalacticFaraday/data/Faraday/catalog_versions/master_catalog_vercustom_large_1param_large_scales_disk_off.fits', overwrite=True)
    hdu.close()

    



if __name__ == '__main__':
    # print a RuntimeWarning  in case of underflows
    np.seterr(all='raise')
    # set seed
    seed = 1000
    ift.random.push_sseq_from_seed(seed)
    run_inference()
