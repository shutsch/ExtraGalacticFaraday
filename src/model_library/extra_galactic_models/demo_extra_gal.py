import nifty7 as ift


def build_demo_extra_gal(domain, mu_a, sigma_a, sigma_b, mu_b):
    # This is a completely cooked up extra-galactic RM model for illustrative purposes only.
    # The model is RM_egal = e**(sigma_a * \xi_a + \mu_a) - e**(sigma_b * \xi_b + \mu_b)/(ln(1 + e^z)),
    # where the sigmas and mus are a hyper-parameters of the model, xi_a and  xi_b are the fields to be determined and z
    # is a number, also to be learned

    # defining the fields

    chi_a = ift.FieldAdapter(domain, 'chi_a')
    chi_b = ift.FieldAdapter(domain, 'chi_b')
    a = (sigma_a*chi_a + mu_a).exp()
    b = (sigma_b*chi_b + mu_b).exp()

    z = ift.FieldAdapter(ift.DomainTuple.scalar_domain(),  'z')

    sky = a - b/z.log1p()

    return {'sky': sky, 'components': {'a': a, 'b': b, 'z': z}, 'amplitudes': {}}
