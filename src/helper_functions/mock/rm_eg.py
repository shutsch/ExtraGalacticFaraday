import nifty8 as ift
import libs as Egf 
import numpy as np



def rm_eg(params, emodel, egal_mock_position, e_z_orig, e_F_orig_at_z, e_z, e_F):

    np.random.seed(seed=params['params_inference.seed'])
    rand_rm=np.random.normal(0.0, 1.0,len(e_z))
    egal_contr = emodel.get_model().sqrt()(egal_mock_position).val*rand_rm
    print('Eg std',np.std(egal_contr))
    print('Eg mean',np.mean(egal_contr))
    Egf.plot_rmeg(params, e_z, e_F, egal_contr, e_z_orig, e_F_orig_at_z, figname='Luminosity_and_z_dependence.png')
    
    return egal_contr
