import nifty7 as ift


def minimization(likelihood, controllers, minimization_dict, kl_type):


    # build the Hamiltonian

    hamiltonian = ift.StandardHamiltonian(likelihood, controller_dict['sampler'])

    # build the minimization functional

    minimizer = ift.NewtonCG(controller_dict['minimizer'])

    kl_dict = minimization_dict[kl_type]
    kl_dict.upgrade({
        'position': ift.from_random(hamiltonian.domain),
        })
    if kltype == 'GeoMetricKL':
        kl_dict.upgrade({'minimizer_samp': minimization_dict['']})

    kl = getattr(ift, kl_type)(**kl_dict)

    for i in range(minimization_dict['global_iterations']):
        kl = minimizer(kl)






            



