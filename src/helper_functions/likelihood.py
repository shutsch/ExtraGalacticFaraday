import numpy as np
import nifty8 as ift
import libs as Egf


def get_implicit_likelihood(params,sky_domain, gal_data_domain, data, z_indices, gal_rm, gal_stddev, galactic_model):
    # build the implicit likelihood for the galactic data points
    implicit_response = Egf.SkyProjector(theta=data['theta'][~z_indices], phi=data['phi'][~z_indices],
                                        domain=sky_domain, target=gal_data_domain) 


    #to use when inference on the noise factors is necessary
    if params['params_inference.eta']==True:
    # Possible all sky variation of alpha, requires pygedm package 
        alpha = 2.5
        log_ymw = np.log(Egf.load_ymw_sky('ymw16', 'mc', params))
        log_ymw /= log_ymw.min()
        log_ymw *= 5
        alpha = implicit_response(ift.Field(ift.makeDomain(implicit_response.domain), log_ymw)).val

        implicit_noise = Egf.SimpleVariableNoise(gal_data_domain, alpha=alpha, q='mode', noise_cov=gal_stddev**2)
        implicit_noise_model=implicit_noise.get_model()
    # build the full model and connect it to the likelihood

        implicit_model = implicit_response @ galactic_model.get_model()
        residual = ift.Adder(-gal_rm) @ implicit_model
        new_dom = ift.MultiDomain.make({'icov': implicit_noise_model.target, 'residual': residual.target})
        n_res = ift.FieldAdapter(new_dom, 'icov')(implicit_noise_model.reciprocal()) + \
            ift.FieldAdapter(new_dom, 'residual')(residual)
        implicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=gal_data_domain, residual_key='residual',
                                                            inverse_covariance_key='icov',
                                                                sampling_dtype=np.dtype(np.float64)) @ n_res

    else:
    #to use with perfect noise knowledge
        implicit_noise = Egf.StaticNoise(gal_data_domain, gal_stddev**2, True)
        implicit_noise_model = None


        # build the full model and connect it to the likelihood    


        implicit_model = implicit_response @ galactic_model.get_model()
        residual = ift.Adder(-gal_rm) @ implicit_model
        implicit_likelihood = ift.GaussianEnergy(inverse_covariance=implicit_noise.get_model(),
                                            sampling_dtype=float) @ residual
    return implicit_likelihood, implicit_noise, implicit_noise_model

           


def get_explicit_likelihood(sky_domain, egal_data_domain, data, z_indices, egal_rm, egal_inverse_noise, galactic_model):
    # build the explicit likelihood for the extra-galactic data points
    explicit_response = Egf.SkyProjector(theta=data['theta'][z_indices], phi=data['phi'][z_indices],
                                            domain=sky_domain, target=egal_data_domain) 
    #if we are not interested in the RM but only in its sigma we do not need to include the Rm in the following line
    explicit_model = explicit_response @ galactic_model.get_model()
    residual = ift.Adder(-egal_rm) @ explicit_model

    new_dom = ift.MultiDomain.make({'icov': egal_inverse_noise.target, 'residual': residual.target})
    n_res = ift.FieldAdapter(new_dom, 'icov')(egal_inverse_noise) + ift.FieldAdapter(new_dom, 'residual')(residual)
    #we need to use the VariableCovarianceGaussianEnerg instead than the GaussianEnergy because the variance (that now
    #includes the eg part that now we are fitting) is varying, is not anymore a costant. When we will include the 
    #correlated eg component we will need to use again the GaussianEnergy. 
    explicit_likelihood = ift.VariableCovarianceGaussianEnergy(domain=egal_data_domain, residual_key='residual',
                                                            inverse_covariance_key='icov',                              
                                                            sampling_dtype=np.dtype(np.float64)) @ n_res
    return explicit_likelihood


       