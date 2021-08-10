import nifty7 as ift


def build_static_noise(domain, noise_cov):
    if isinstance(noise_cov, ift.Field):
        return ift.makeOp(noise_cov)
    return ift.makeOp(ift.Field(domain, noise_cov))
