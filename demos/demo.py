import nifty7 as ift


# Type checking

def run_inference():

    # build the full model and connect it to the likelihood

    full_model = response_op @ galactic_model + extra_galactic_model

    likelihood = likelihood @ full_model
