import nifty8 as ift
import libs as Egf
import numpy as np
from nifty_cmaps import ncmap
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
    scatter_pairs = {'intrinsic': (ecomponents['chi_lum'], ecomponents['chi_int_0']),'environmental': (ecomponents['chi_red'], ecomponents['chi_env_0'])}

    plotting_kwargs = {'faraday_sky': {'cmap': 'fm', 'cmap_stddev': 'fu', 
                                       'vmin_mean':'-250', 'vmax_mean':'250', 
                                       'vmin_std':'0', 'vmax_std':'80'},
                       'intrinsic': {'x_label': 'chi_lum', 'y_label': 'sigma_int_0'},
                       'environmental': {'x_label': 'chi_red', 'y_label': 'sigma_env_0'}}

  
    samples = ift.ResidualSampleList.load('samples_posterior')

    m, v = samples.sample_stat(sky_models['faraday_sky'])

    gal_rm_mock=implicit_response(m)
    gal_rm_var_mock=implicit_response(v)

    egal_rm_mock=explicit_response(m)
    egal_rm_var_mock=explicit_response(v)


    gal_gal_residual=(gal_rm - gal_rm_mock)/gal_rm_var_mock.sqrt()
    egal_gal_residual=(egal_rm - egal_rm_mock)/egal_rm_var_mock.sqrt()




    plo=ift.Plot()
    plo.add(m, cmap=getattr(ncmap, 'fm')(), title='Mean', vmin=-45000, vmax=45000)
    plo.add(v.sqrt(), cmap=getattr(ncmap, 'fu')(), title='Std', vmin=0.0, vmax=45000)
    plo.add(implicit_response.adjoint(gal_gal_residual), cmap=getattr(ncmap, 'pm')(), vmin=-0.5, vmax=0.5, title='Gal Galactic residuals')
    plo.add(explicit_response.adjoint(egal_gal_residual), cmap=getattr(ncmap, 'pm')(), vmin=-0.25, vmax=0.25, title='EG Galactic residulas')
    plo.add(implicit_response.adjoint(gal_rm_var_mock.sqrt()), cmap=getattr(ncmap, 'pm')(), vmax=10000, title='Gal points Rec RM std')
    plo.add(explicit_response.adjoint(egal_rm_var_mock.sqrt()), cmap=getattr(ncmap, 'pm')(), vmax=2500, title='EG points Rec RM std')
    plo.output(nx=2, ny=3, xsize=2 * 4, ysize=3 * 4, name='Residuals_sky_distribution.png')


    fig, axs = plt.subplots(3, 4)
    fig.tight_layout()
    axs[0,0].plot(gal_gal_residual.val, 'k.')
    axs[0,2].plot(egal_gal_residual.val, 'k.')
    axs[0,0].set_xlabel('Point number')
    axs[0,0].set_ylabel('Gal Galactic residual')
    axs[0,2].set_xlabel('Point number')
    axs[0,2].set_ylabel('EG Galactic residual')

    axs[0,1].hist(np.abs(gal_gal_residual.val),bins='auto', density=True)
    axs[0,1].set_xlabel('Gal Galactic residual')
    axs[0,1].set_ylim(0,3)
    axs[0,1].set_xscale('log')
    axs[0,3].hist(egal_gal_residual.val,bins='auto', density=True)
    axs[0,3].set_xlabel('EG Galactic residual')
    axs[0,3].set_xscale('log')
    axs[0,3].set_xlim(1e-3, 1e1)

    axs[1,0].scatter(gal_rm_var_mock.sqrt().val, gal_gal_residual.val)
    axs[1,1].scatter(egal_rm_var_mock.sqrt().val, egal_gal_residual.val)
    axs[1,0].set_xlabel('Rec uncertainty')
    axs[1,0].set_xlim(-10,50000)
    axs[1,1].set_xlim(-10,10000)
    axs[1,0].set_ylabel('Gal Galactic residual')
    axs[1,1].set_xlabel('Rec uncertainty')
    axs[1,1].set_ylabel('EG Galactic residual')
    axs[1,0].axvline(10000, color='red', linestyle='--')
    axs[1,1].axvline(2500, color='red', linestyle='--')

    axs[2,0].scatter(gal_stddev.val, gal_gal_residual.val)
    axs[2,1].scatter(egal_stddev.val, egal_gal_residual.val)
    axs[2,0].set_xlabel('Obs uncertainty')
    axs[2,0].set_ylabel('Gal Galactic residual')
    axs[2,1].set_xlabel('Obs uncertainty')
    axs[2,1].set_ylabel('EG Galactic residual')
    axs[2,1].set_xlim(-10,100)

    axs[1,2].scatter((gal_rm - gal_rm_mock).val, gal_gal_residual.val)
    axs[1,3].scatter((egal_rm - egal_rm_mock).val, egal_gal_residual.val)
    axs[1,2].set_xlabel('Gal (Obs - Reconstr)')
    axs[1,2].set_ylabel('Gal Galactic residual')
    axs[1,3].set_xlabel('EG (Obs - Reconstr)')
    axs[1,3].set_ylabel('EG Galactic residual')
    axs[1,2].set_xlim(-10000,10000)
    axs[1,3].set_xlim(-5000,5000)



    axs[2,2].scatter(gal_rm_mock.val, gal_gal_residual.val)
    axs[2,3].scatter(egal_rm_mock.val, egal_gal_residual.val)
    axs[2,2].set_xlabel('Gal Reconstr')
    axs[2,2].set_ylabel('Gal Galactic residual')
    axs[2,2].set_xlim(-1000,1000)
    axs[2,3].set_xlabel('EG Reconstr')
    axs[2,3].set_ylabel('EG Galactic residual')
    axs[2,3].set_xlim(-500,500)



    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig('Residual.png', bbox_inches='tight')
    plt.show()



   
if __name__ == '__main__':
    # print a RuntimeWarning  in case of underflows
    np.seterr(under='warn') 
    np.seterr(all='raise')
    # set seed
    seed = 1000
    ift.random.push_sseq_from_seed(seed)
    run_inference()
