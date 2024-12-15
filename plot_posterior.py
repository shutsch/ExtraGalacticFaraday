import nifty8 as ift
import libs as Egf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def run_inference():
    


    # set the HealPix resolution parameter and the sky domain

    sky_domain = ift.makeDomain(ift.HPSpace(Egf.config['params']['nside']))

    # load_the data, define domains, covariance and projection operators

    data = Egf.get_rm(filter_pulsars=True, version='custom_eg_large_four_param', default_error_level=0.5)

    # filter
    z_indices = ~np.isnan(data['z_best'])

    e_rm = np.array(data['rm'][z_indices])
    e_stddev = np.array(data['rm_err'][z_indices])
    e_z = np.array(data['z_best'][z_indices])
    e_F = np.array(data['stokesI'][z_indices])

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


    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(e_rm),)))

    egal_rm = ift.Field(egal_data_domain, e_rm)
    egal_stddev = ift.Field(egal_data_domain, e_stddev)
    
    # build the full model and connect it to the likelihood
    # set the extra-galactic model hyper-parameters and initialize the model
    egal_model_params = {'z': e_z, 'F': e_F }
      
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
    explicit_model = explicit_response @ galactic_model.get_model()
    #egal_model = explicit_response @ galactic_model.get_model() + emodel.get_model()
    residual = ift.Adder(-egal_rm) @ explicit_model
    
    new_dom = ift.MultiDomain.make({'icov': egal_inverse_noise.target, 'residual': residual.target})
    n_res = ift.FieldAdapter(new_dom, 'icov')(egal_inverse_noise) + \
        ift.FieldAdapter(new_dom, 'residual')(residual)
    
    #we need to use the VariableCovarianceGaussianEnerg instead than the GaussianEnergy because the variance (that now
    #includes the eg part that now we are fitting) is varying, is not anymore a costant. When we will include the 
    #correlated eg component we will need to use again the GaussianEnergy. 
    explicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=egal_data_domain, residual_key='residual',
                                                               inverse_covariance_key='icov',
                                                               sampling_dtype=np.dtype(np.float64)) @ n_res
    

    g_rm = np.array(data['rm'][~z_indices])
    g_stddev = np.array(data['rm_err'][~z_indices])

    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((len(g_rm),)))

    gal_rm = ift.Field(gal_data_domain, g_rm)
    gal_stddev = ift.Field(gal_data_domain, g_stddev)

    implicit_response = Egf.SkyProjector(theta=data['theta'][~z_indices],
                                         phi=data['phi'][~z_indices],
                                         domain=sky_domain, target=gal_data_domain)



    # Possible all sky variation of alpha, requires pygedm package 
    #alpha = 2.5
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
    
    likelihoods={'implicit_likelihood': implicit_likelihood, 'explicit_likelihood': explicit_likelihood}
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
    

    likelihoods={'implicit_likelihood': implicit_likelihood, 'explicit_likelihood': explicit_likelihood}



    #mean = ift.ResidualSampleList.load_mean('samples_posterior')
    #samples = ift.ResidualSampleList.load('samples_posterior')
    samples = ift.ResidualSampleList.load('/raid/ERG/DEFROST_MOCK/src/ExtraGalacticFaraday/runs/demo/results_zero_centered_prior/pickle/last')

    cr=np.array([s for s in samples.iterator(ecomponents['chi_red'])])
    mr, vr = samples.sample_stat(ecomponents['chi_red'])

    ci0=np.array([s for s in samples.iterator(ecomponents['chi_int_0'])])
    mi0, vi0 = samples.sample_stat(ecomponents['chi_int_0'])

    cl=np.array([s for s in samples.iterator(ecomponents['chi_lum'])])
    ml, vl = samples.sample_stat(ecomponents['chi_lum'])

    ce0=np.array([s for s in samples.iterator(ecomponents['chi_env_0'])])
    me0, ve0 = samples.sample_stat(ecomponents['chi_env_0'])
    
    sr=np.sqrt(vr.val)
    si0=np.sqrt(vi0.val)
    sl=np.sqrt(vl.val)
    se0=np.sqrt(ve0.val)

    print('cr', mr.val, sr)
    print('ci0', mi0.val, si0)
    print('cl', ml.val, sl)
    print('ce0', me0.val, se0)




    cr_list=[]
    cl_list=[]
    ci0_list=[]
    ce0_list=[]
    for i in range(0,len(cr)):
        cr_list.append(cr[i].val)
        cl_list.append(cl[i].val)
        ci0_list.append(ci0[i].val)
        ce0_list.append(ce0[i].val)
    cr_array=np.array(cr_list)
    cl_array=np.array(cl_list)
    ci0_array=np.array(ci0_list)
    ce0_array=np.array(ce0_list)

 
    fig, axs = plt.subplots(3, 3)
    
    axs[0,0].scatter(cr_array, ci0_array, color='k')
    axs[0,0].set_ylabel('$\chi_{int,0}$')


    axs[0,1].scatter(cl_array, ci0_array, color='k')
    axs[0,1].set_yticklabels([])

    axs[0,2].scatter(ce0_array, ci0_array, color='k')
    axs[0,2].set_xlabel('$\chi_{env,0}$')
    axs[0,2].set_yticklabels([])



    axs[1,0].scatter(cr_array, ce0_array, color='k')
    axs[1,0].set_ylabel('$\chi_{env,0}$')

    axs[1,1].scatter(cl_array, ce0_array, color='k')
    axs[1,1].set_xlabel('$\chi_{lum}$')
    axs[1,1].set_yticklabels([])

    axs[1,2].axis('off')

    axs[2,0].scatter(cr_array, cl_array, color='k')
    axs[2,0].set_xlabel('$\chi_{red}$')
    axs[2,0].set_ylabel('$\chi_{lum}$')


    axs[2,1].axis('off')
    axs[2,2].axis('off')


    ellipse1_1sigma = Ellipse(xy=(mr.val, mi0.val), width=1*2*sr, height=1*2*si0, edgecolor='green', fc='None', lw=2)
    ellipse1_2sigma = Ellipse(xy=(mr.val, mi0.val), width=2*2*sr, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
    ellipse1_3sigma = Ellipse(xy=(mr.val, mi0.val), width=3*2*sr, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
    axs[0,0].add_patch(ellipse1_1sigma)
    axs[0,0].add_patch(ellipse1_2sigma)
    axs[0,0].add_patch(ellipse1_3sigma)


    ellipse2_1sigma = Ellipse(xy=(ml.val, mi0.val), width=1*2*sl, height=1*2*si0, edgecolor='green', fc='None', lw=2)
    ellipse2_2sigma = Ellipse(xy=(ml.val, mi0.val), width=2*2*sl, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
    ellipse2_3sigma = Ellipse(xy=(ml.val, mi0.val), width=3*2*sl, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
    axs[0,1].add_patch(ellipse2_1sigma)
    axs[0,1].add_patch(ellipse2_2sigma)
    axs[0,1].add_patch(ellipse2_3sigma)

    ellipse3_1sigma = Ellipse(xy=(me0.val, mi0.val), width=1*2*se0, height=1*2*si0, edgecolor='green', fc='None', lw=2)
    ellipse3_2sigma = Ellipse(xy=(me0.val, mi0.val), width=2*2*se0, height=2*2*si0, edgecolor='cyan', fc='None', lw=2)
    ellipse3_3sigma = Ellipse(xy=(me0.val, mi0.val), width=3*2*se0, height=3*2*si0, edgecolor='blue', fc='None', lw=2)
    axs[0,2].add_patch(ellipse3_1sigma)
    axs[0,2].add_patch(ellipse3_2sigma)
    axs[0,2].add_patch(ellipse3_3sigma)



    ellipse4_1sigma = Ellipse(xy=(mr.val, me0.val), width=1*2*sr, height=1*2*se0, edgecolor='green', fc='None', lw=2)
    ellipse4_2sigma = Ellipse(xy=(mr.val, me0.val), width=2*2*sr, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
    ellipse4_3sigma = Ellipse(xy=(mr.val, me0.val), width=3*2*sr, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
    axs[1,0].add_patch(ellipse4_1sigma)
    axs[1,0].add_patch(ellipse4_2sigma)
    axs[1,0].add_patch(ellipse4_3sigma)


    ellipse5_1sigma = Ellipse(xy=(ml.val, me0.val), width=1*2*sl, height=1*2*se0, edgecolor='green', fc='None', lw=2)
    ellipse5_2sigma = Ellipse(xy=(ml.val, me0.val), width=2*2*sl, height=2*2*se0, edgecolor='cyan', fc='None', lw=2)
    ellipse5_3sigma = Ellipse(xy=(ml.val, me0.val), width=3*2*sl, height=3*2*se0, edgecolor='blue', fc='None', lw=2)
    axs[1,1].add_patch(ellipse5_1sigma)
    axs[1,1].add_patch(ellipse5_2sigma)
    axs[1,1].add_patch(ellipse5_3sigma)



    ellipse6_1sigma = Ellipse(xy=(mr.val, ml.val), width=1*2*sr, height=1*2*sl, edgecolor='green', fc='None', lw=2)
    ellipse6_2sigma = Ellipse(xy=(mr.val, ml.val), width=2*2*sr, height=2*2*sl, edgecolor='cyan', fc='None', lw=2)
    ellipse6_3sigma = Ellipse(xy=(mr.val, ml.val), width=3*2*sr, height=3*2*sl, edgecolor='blue', fc='None', lw=2)
    axs[2,0].add_patch(ellipse6_1sigma)
    axs[2,0].add_patch(ellipse6_2sigma)
    axs[2,0].add_patch(ellipse6_3sigma)

    ellipse_prior1 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
    ellipse_prior2 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
    ellipse_prior3 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*1.0, edgecolor='red', fc='None', lw=1)
    ellipse_prior4 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)
    ellipse_prior5 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)
    ellipse_prior6 = Ellipse(xy=(0.0, 0.0), width=2*1.0, height=2*3.0, edgecolor='red', fc='None', lw=1)

    axs[2,0].add_patch(ellipse_prior1)
    axs[1,0].add_patch(ellipse_prior2)
    axs[1,1].add_patch(ellipse_prior3)
    axs[0,2].add_patch(ellipse_prior4)
    axs[0,0].add_patch(ellipse_prior5)
    axs[0,1].add_patch(ellipse_prior6)

    plt.subplots_adjust(wspace=0, hspace=0)



    axs[0,0].axhline(y = 5.0, color = 'b', linestyle = '--') 
    axs[0,0].axvline(x = -0.5, color = 'b', linestyle='--')

    axs[1,0].axhline(y = 0.0, color = 'b', linestyle = '--') 
    axs[1,0].axvline(x = -0.5, color = 'b', linestyle='--')


    axs[2,0].axhline(y = 0.0, color = 'b', linestyle = '--') 
    axs[2,0].axvline(x = -0.5, color = 'b', linestyle='--')

    axs[0,1].axhline(y = 5.0, color = 'b', linestyle = '--') 
    axs[0,1].axvline(x = 0.0, color = 'b', linestyle='--')


    axs[1,1].axhline(y = 0.0, color = 'b', linestyle = '--') 
    axs[1,1].axvline(x = 0.0, color = 'b', linestyle='--')

    axs[0,2].axhline(y = 5.0, color = 'b', linestyle = '--') 
    axs[0,2].axvline(x = 0.0, color = 'b', linestyle='--')

    plt.savefig('EG_posterior.png', bbox_inches='tight')

    plt.show()
   



if __name__ == '__main__':
    # print a RuntimeWarning  in case of underflows
    np.seterr(all='raise')
    # set seed
    seed = 1000
    ift.random.push_sseq_from_seed(seed)
    run_inference()
