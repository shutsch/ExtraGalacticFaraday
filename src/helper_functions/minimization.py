import nifty7 as ift


def minimization(likelihood, controllers, minimization_dict, kl_type):

    # build the Hamiltonian

    hamiltonian = ift.StandardHamiltonian(likelihood, controllers['sampler'])

    # build the minimization functional

    minimizer = ift.NewtonCG(controllers['minimizer'])

    kl_dict = minimization_dict[kl_type]
    kl_dict.upgrade({
        'position': ift.from_random(hamiltonian.domain),
        })
    if kl_type == 'GeoMetricKL':
        kl_dict.upgrade({'minimizer_samp': minimization_dict['']})

    kl = getattr(ift, kl_type)(**kl_dict)

    for i in range(minimization_dict['global_iterations']):
        kl = minimizer(kl)






            



