import nifty8 as ift
import libs as Egf
import numpy as np
from astropy.io import fits
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

    egal_rm = np.array(data['rm'][z_indices])
    egal_stddev = np.array(data['rm_err'][z_indices])
    egal_z = np.array(data['z_best'][z_indices])
    egal_L = np.array(data['stokesI'][z_indices])

    # set the sky model hyper-parameters and initialize the Faraday 2020 sky model
    #new parameters given by Sebastian
    log_amplitude_params = {'fluctuations': {'asperity': None, 
                                             'flexibility': [1., 1.],  
                                             'fluctuations': [1.0, 0.5], 
                                             'loglogavgslope': [-11./3, 2.],},
                          'offset': {'offset_mean': 4., 
                                     'offset_std': [1., 1.]},}

    sign_params = {'fluctuations': {'asperity': None, 
                                    'flexibility': [1., 1.], 
                                    'fluctuations': [5.0, 4.0], 
                                    'loglogavgslope': [-11./3, 2.], },
                   'offset': {'offset_mean': 0, 
                              'offset_std': [5., 4.]},}
    
   


    galactic_model = Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(egal_rm),)))

    egal_rm = ift.Field(egal_data_domain, egal_rm)
    egal_stddev = ift.Field(egal_data_domain, egal_stddev)
    

    

    # build the full model and connect it to the likelihood
    # set the extra-galactic model hyper-parameters and initialize the model
    egal_model_params = {'z': egal_z,'L': egal_L, 
         }
      
    emodel = Egf.ExtraGalDemoModel(egal_data_domain, egal_model_params)

    #if we are not interested in the RM but only in its sigma we can consider the eg sigma as a noise and sum the two here. 
    #we include it here but not in the Variable Noise below because the variable noise include the eta factors and applies only to
    #the Tayolor catalog. Here we are considering the LOFAR catalog. When we will include the correlated eg component, the line 
    #below will include again only the noise. 
    

    noise_params = {
        'egal_var': egal_stddev**2,
        'emodel': emodel.get_model()
    }

    egal_inverse_noise = Egf.EgalAddingNoise(egal_data_domain, noise_params, inverse=True).get_model()


    
    explicit_response = Egf.SkyProjector(theta=data['theta'][z_indices], phi=data['phi'][z_indices],
                                         domain=sky_domain, target=egal_data_domain) 

      
    #if we are not interested in the RM but only in its sigma we do not need to include the Rm in the following line
    egal_model = explicit_response @ galactic_model.get_model()
    #egal_model = explicit_response @ galactic_model.get_model() + emodel.get_model()
    residual = ift.Adder(-egal_rm) @ egal_model
    #we need to use the VariableCovarianceGaussianEnerg instead than the GaussianEnergy because the variance (that now
    #includes the eg part that now we are fitting) is varying, is not anymore a costant. When we will include the 
    #correlated eg component we will need to use again the GaussianEnergy. 
    new_dom = ift.MultiDomain.make({'icov': egal_inverse_noise.target, 'residual': residual.target})
    n_res = ift.FieldAdapter(new_dom, 'icov')(egal_inverse_noise) + \
        ift.FieldAdapter(new_dom, 'residual')(residual)
    explicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=egal_data_domain, residual_key='residual',
                                                               inverse_covariance_key='icov',
                                                               sampling_dtype=np.dtype(np.float64)) @ n_res
    

    gal_rm = np.array(data['rm'][~z_indices])
    gal_stddev = np.array(data['rm_err'][~z_indices])

  
    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(gal_rm),)))

    gal_rm = ift.Field(gal_data_domain, gal_rm)
    gal_stddev = ift.Field(gal_data_domain, gal_stddev)

    implicit_response = Egf.SkyProjector(theta=data['theta'][~z_indices],
                                         phi=data['phi'][~z_indices],
                                         domain=sky_domain, target=gal_data_domain)


    alpha = 2.5

    # Possible all sky variation of alpha, requires pygedm package 
    
    #log_ymw = np.log(Egf.load_ymw_sky('./data/', nside=Egf.config['params']['nside'], model='ymw16', mode='mc'))
    #log_ymw /= log_ymw.min()
    #log_ymw *= 5
    #alpha = implicit_response(ift.Field(ift.makeDomain(implicit_response.domain), log_ymw)).val


    #implicit_noise = Egf.SimpleVariableNoise(gal_data_domain, alpha=alpha, q='mode', noise_cov=gal_stddev**2).get_model()
    implicit_noise = Egf.StaticNoise(gal_data_domain, gal_stddev**2, True)

    # build the full model and connect it to the likelihood

    implicit_model = implicit_response @ galactic_model.get_model()
    residual = ift.Adder(-gal_rm) @ implicit_model
    #new_dom = ift.MultiDomain.make({'icov': implicit_noise.target, 'residual': residual.target})
    #n_res = ift.FieldAdapter(new_dom, 'icov')(implicit_noise.reciprocal()) + \
    #    ift.FieldAdapter(new_dom, 'residual')(residual)
    #implicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=gal_data_domain, residual_key='residual',
    #                                                           inverse_covariance_key='icov',
    #                                                           sampling_dtype=np.dtype(np.float64)) @ n_res
    implicit_likelihood = ift.GaussianEnergy(inverse_covariance=implicit_noise.get_model(),
                                             sampling_dtype=float) @ residual

    # set run parameters and start the inference
    components = galactic_model.get_components()
    ecomponents = emodel.get_components()

    sky_models = {'faraday_sky': galactic_model.get_model(), 'profile': components['log_profile'].exp(),
                  'sign': components['sign']}
    power_models = {'log_profile': components['log_profile_amplitude'], 'sign': components['sign_amplitude']}
   
    #the value that we plot are indeed the values in the position field 
    #scatter_pairs = {'intrinsic': (ecomponents['chi_lum'], ecomponents['chi_int_0']),'environmental': (ecomponents['chi_red'], ecomponents['chi_env_0'])}

    plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
                                       'vmin_mean':'-250', 'vmax_mean':'250', 
                                       'vmin_std':'0', 'vmax_std':'80'},
                       'intrinsic': {'x_label': 'chi_lum', 'y_label': 'sigma_int_0'},
                       'environmental': {'x_label': 'chi_red', 'y_label': 'sigma_env_0'}}

    # Specify noise
    noise = 0.05
    N = ift.ScalingOperator(ift.UnstructuredDomain(len(data['theta'])), noise, np.float64)

    overall_response = Egf.SkyProjector(theta=data['theta'],
                                         phi=data['phi'],
                                         domain=sky_domain, target=ift.UnstructuredDomain(len(data['theta'])))

    overall_model= overall_response @ galactic_model.get_model()

    mock_position = ift.from_random(overall_model.domain, 'normal')
    overall_data=overall_model(mock_position)

    egal_mock_position = ift.from_random(emodel.get_model().domain, 'normal')
    data_array=np.array(overall_data.val)
    data_array[z_indices]+=emodel.get_model()(egal_mock_position).val
    
    data_field=ift.makeField(ift.UnstructuredDomain(len(data['theta'])), data_array)

    data_RM = data_field + N.draw_sample()

    sign = components['sign'].force(mock_position)

    #Plot 1
    plot = ift.Plot()
    plot.add(overall_response.adjoint(overall_data), vmin=-250, vmax=250)
    #plot.add(overall_response.adjoint(overall_data), vmin=min(sign.val)-10, vmax=max(sign.val)+10, cmap='fm')
    plot.add(overall_response.adjoint(data_RM), vmin=min(sign.val)-10, vmax=max(sign.val)+10)
    plot.add(sign, vmin=min(sign.val)-10, vmax=max(sign.val)+10)
    plot.output()

    #Plot 2
    #fig, axs = plt.subplots(1, 2)

    #axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
    #axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
    #axs[1].scatter(data['rm'][z_indices],data_RM.val[z_indices])

    #axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
    #axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
    #axs[0].scatter(data['rm'][~z_indices],data_RM.val[~z_indices])
    #plt.show()




    data['rm'] = np.array(data_RM.val)
    data['rm_err'] =  noise*np.ones(np.array(data_RM.val).size)
    
    #hdu= fits.open('/home/valentina/Documents/PROJECTS/BAYESIAN_CODE/DATA/new_catalog.fits')
    #hdu[1].data['rm'][np.where(hdu[1].data['type']!='Pulsar')] = np.array(data_RM.val)
    #hdu[1].data['rm_err'][np.where(hdu[1].data['type']!='Pulsar')] =  noise*np.ones(np.array(data_RM.val).size)
    #hdu.writeto('/home/valentina/Documents/PROJECTS/BAYESIAN_CODE/DEFROST_LAST/ExtraGalacticFaraday/data/Faraday/catalog_versions/master_catalog_vercustom_prior.fits', overwrite=True)
    #hdu.close()


if __name__ == '__main__':
    # print a RuntimeWarning  in case of underflows
    np.seterr(under='warn') 
    np.seterr(all='raise')
    # set seed
    seed = 1000
    ift.random.push_sseq_from_seed(seed)
    run_inference()
