import nifty7 as ift


def mock_extra_gal(param_dict, hyperparameters):
    # This is a completely cooked up extra-galactic RM model for illustrative purposes only.
    # The model is RM_egal = e**(sigma_a * \xi_a + \mu_a) - e**(sigma_b * \xi_b + \mu_b)/(ln(1 + e^z)),
    # where the sigmas and mus are a hyper-parameters of the model, xi_a and  xi_b are the fields to be determined and

    domain = param_dict['n_sources']

    # getting the hyper-parameters

    sigma_a = hyperparameters['sigma_a']
    mu_a = hyperparameters['mu_a']
    sigma_b = hyperparameters['sigma_b']
    mu_b = hyperparameters['mu_b']

    # defining the fields

    chi_a = ift.FieldAdapter(domain, 'chi_a')
    chi_b = ift.FieldAdapter(domain, 'chi_b')
    a = (sigma_a*chi_a + mu_a).exp()
    b = (sigma_b*chi_b + mu_b).exp()

    z = ift.FieldAdapter(ift.DomainTuple.scalar_domain(),  'z')

    sky = a - b/z.log1p()

    return {'sky': sky, 'components': {'a': a, 'b': b, 'z': z}, 'amplitudes': {}}
