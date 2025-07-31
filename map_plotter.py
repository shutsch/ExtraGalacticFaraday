import numpy as np
import nifty8 as ift
import libs as Egf 
import matplotlib.pyplot as plt
import matplotlib
import math as m
from matplotlib import cm
from astropy.cosmology import FlatLambdaCDM
from src.helper_functions.misc import gal2gal
matplotlib.use('TkAgg')

#cosmo and constants

light_speed =  Egf.const['c']
h =  Egf.const['Jens']['h']
Wm = Egf.const['Jens']['Wm']
Wc = Egf.const['Jens']['Wc']
Wl = Egf.const['Jens']['Wl']
H0 = 100 * h
    
cosmo = FlatLambdaCDM(H0=H0, Om0=Wm)  
L0 = float(Egf.const['L0'])
D0 = Egf.const['D0']
factor = float(Egf.const['factor'])



class Map_Plotter():
    def __init__(self, args):
        self.ecomponents = args['ecomponents']
        self.params = args['params']

    def plot(self, figname):
        params= self.params

        samples = ift.ResidualSampleList.load(f'{self.params["params_inference.results_path"]}pickle/last')

        cr=np.array([s for s in samples.iterator(self.ecomponents['chi_red'])])
        mr, vr = samples.sample_stat(self.ecomponents['chi_red'])

        ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])

        cl=np.array([s for s in samples.iterator(self.ecomponents['chi_lum'])])
        ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        
        sr=np.sqrt(vr.val)
        si0=np.sqrt(vi0.val)
        sl=np.sqrt(vl.val)
        se0=np.sqrt(ve0.val)

        print('cr', mr.val, 'pm', sr)
        print('ci0', mi0.val, 'pm', si0)
        print('cl', ml.val, 'pm', sl)
        print('ce0', me0.val, 'pm', se0)

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

        
        sky_domain = ift.makeDomain(ift.HPSpace(params['params_map.nside']))
        catalog_version = 'custom_sim'

        data = Egf.get_rm(filter_pulsars=True, version=f'{catalog_version}', default_error_level=0.5, params=params)
        z_indices = ~np.isnan(data['z_best'])


        e_z = np.array(data['z_best'][z_indices])
        Dl=cosmo.luminosity_distance(e_z).value

        e_F = np.array(data['stokesI'][z_indices])
        e_L=e_F*4*m.pi*Dl**2*factor

        eg_l = np.array(data['l'][z_indices])
        eg_b = np.array(data['b'][z_indices])

        theta_eg, phi_eg = gal2gal(eg_l, eg_b) # converting to colatitude and logitude in radians

    
        lthetaeg = len(theta_eg)
        
        eg_projector = Egf.SkyProjector(ift.makeDomain(ift.HPSpace(self.params['params_map.nside'])), ift.makeDomain(ift.UnstructuredDomain(lthetaeg)), theta=theta_eg, phi=phi_eg)


        egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lthetaeg)))

        # build the full model and connect it to the likelihood
        # set the extra-galactic model hyper-parameters and initialize the model
        egal_model_params = {'z': e_z, 'F': e_F, 'params': params}
        
        emodel = Egf.ExtraGalModel(egal_data_domain, egal_model_params)

        #egal_mock_position = ift.from_random(emodel.get_model().domain, 'normal')
        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)
        epd = egal_mock_position.to_dict() 
        chiint0_field = epd['chi_int_0'] 
        chienv0_field = epd['chi_env_0'] 
        chilum_field = epd['chi_lum'] 
        chired_field = epd['chi_red'] 
        epd['chi_int_0'] = ift.full(chiint0_field.domain, float(mi0.val)) 
        epd['chi_env_0'] = ift.full(chienv0_field.domain, float(me0.val)) 
        epd['chi_lum'] = ift.full(chilum_field.domain, float(ml.val)) 
        epd['chi_red'] = ift.full(chired_field.domain, float(mr.val)) 
        egal_mock_position = egal_mock_position.from_dict(epd)
        egal_contr = emodel.get_model().sqrt()(egal_mock_position).val
        eg=ift.makeField(ift.UnstructuredDomain(lthetaeg), egal_contr)

        egal_mock_position = ift.full(emodel.get_model().domain, 0.0)
        epd = egal_mock_position.to_dict() 
        chiint0_field = epd['chi_int_0'] 
        chienv0_field = epd['chi_env_0'] 
        chilum_field = epd['chi_lum'] 
        chired_field = epd['chi_red'] 
        epd['chi_int_0'] = ift.full(chiint0_field.domain, float(params['mean.mean_int'])) 
        epd['chi_env_0'] = ift.full(chienv0_field.domain, float(params['mean.mean_env'])) 
        epd['chi_lum'] = ift.full(chilum_field.domain, float(params['mean.mean_lum'])) 
        epd['chi_red'] = ift.full(chired_field.domain, float(params['mean.mean_red'])) 
        egal_mock_position_mock = egal_mock_position.from_dict(epd)
        egal_contr_mock = emodel.get_model().sqrt()(egal_mock_position_mock).val
        eg_mock=ift.makeField(ift.UnstructuredDomain(lthetaeg), egal_contr_mock)

        print(f'mock:{egal_mock_position.val}')
        plot = ift.Plot()
        plot.add(eg_projector.adjoint(eg_mock), vmin=-50, vmax=50, title='Ground truth')
        plot.add(eg_projector.adjoint(eg), vmin=-50, vmax=50, title='Posterior')
        plot.output(name=f'{self.params["params_inference.plot_path"]}{figname}')