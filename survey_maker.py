import nifty8 as ift
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from src.helper_functions.data.rmtable import read_FITS
from src.helper_functions.misc import gal2gal
from src.helper_functions.parameters_maker import Parameters_maker
import matplotlib.pyplot as plt
import healpy as hp


class SurveyMaker():
  
    def __init__(self, params):
        self.params = params

    def make_survey(self): 
        params=self.params 

        if params['params_mock_cat.maker_params.surveys.make_survey2'] == True:
            size_cat = params['params_mock_cat.maker_params.surveys.los'] + params['params_mock_cat.maker_params.surveys.los2']
        else:
            size_cat = params['params_mock_cat.maker_params.surveys.los'] 

        pos_err=np.empty(size_cat)
        rm=np.empty(size_cat)
        rm_err=np.empty(size_cat)
        rm_width=np.empty(size_cat)
        rm_width_err=np.empty(size_cat)
        complex_flag=np.empty(size_cat)
        complex_test=np.empty(size_cat)
        rm_method=np.empty(size_cat)
        ionosphere=np.empty(size_cat)
        Ncomp=np.empty(size_cat)
        stokesI=np.empty(size_cat)
        stokesI_err=np.empty(size_cat)
        spectral_index=np.empty(size_cat)
        spectral_index_err=np.empty(size_cat)
        reffreq_I=np.empty(size_cat)
        polint=np.empty(size_cat)
        polint_err=np.empty(size_cat)
        pol_bias=np.empty(size_cat)
        flux_type=np.empty(size_cat)
        aperture=np.empty(size_cat)
        fracpol=np.empty(size_cat)
        fracpol_err=np.empty(size_cat)
        polangle=np.empty(size_cat)
        polangle_err=np.empty(size_cat)
        reffreq_pol=np.empty(size_cat)
        stokesQ=np.empty(size_cat)
        stokesQ_err=np.empty(size_cat)
        stokesU=np.empty(size_cat)
        stokesU_err=np.empty(size_cat)
        derot_polangle=np.empty(size_cat)
        derot_polangle_err=np.empty(size_cat)
        stokesV=np.empty(size_cat)
        stokesV_err=np.empty(size_cat)
        beam_maj=np.empty(size_cat)
        beam_min=np.empty(size_cat)
        beam_pa=np.empty(size_cat)
        reffreq_beam=np.empty(size_cat)
        minfreq=np.empty(size_cat)
        maxfreq=np.empty(size_cat)
        channelwidth=np.empty(size_cat)
        Nchan=np.empty(size_cat)
        rmsf_fwhm=np.empty(size_cat)
        noise_chan=np.empty(size_cat)
        telescope=np.empty(size_cat)
        int_time=np.empty(size_cat)
        epoch=np.empty(size_cat)
        interval=np.empty(size_cat)
        leakage=np.empty(size_cat)
        beamdist=np.empty(size_cat)
        catalog=np.empty(size_cat, dtype='S10')
        dataref=np.empty(size_cat)
        cat_id=np.empty(size_cat)
        types=np.empty(size_cat)
        notes=np.empty(size_cat)
        z_best=np.empty(size_cat)



        s_space = ift.HPSpace(params['params_inference.nside'])
        s_space_domain = ift.makeDomain(s_space)   

        phi=np.random.uniform(params['params_mock_cat.maker_params.surveys.phi_low']*np.pi,params['params_mock_cat.maker_params.surveys.phi_high']*np.pi,params['params_mock_cat.maker_params.surveys.los'])
        ra_deg1=phi*180.0/np.pi

        p_low= (np.sin(np.radians(params['params_mock_cat.maker_params.surveys.delta_low']))+1.0)/2.0
        p_high= (np.sin(np.radians(params['params_mock_cat.maker_params.surveys.delta_high']))+1.0)/2.0
        p=np.random.uniform(p_low,p_high,params['params_mock_cat.maker_params.surveys.los'])

        sin_delta=2*p-1
        delta=np.arcsin(sin_delta)
        dec_deg1=np.degrees(delta)

        c = SkyCoord(ra=ra_deg1*u.degree, dec=dec_deg1*u.degree, frame='icrs')
        l1=c.galactic.l.value
        b1=c.galactic.b.value
        survey_indices= hp.ang2pix(params['params_inference.nside'], l1, b1, lonlat=True)
        all_indices=np.zeros(12*params['params_inference.nside']**2)
        all_indices[survey_indices]=1
        catalog[0:params['params_mock_cat.maker_params.surveys.los']]=params['params_mock_cat.maker_params.surveys.name']


        if params['params_mock_cat.maker_params.surveys.make_survey2'] == True:
            phi2=np.random.uniform(params['params_mock_cat.maker_params.surveys.phi_low2']*np.pi,params['params_mock_cat.maker_params.surveys.phi_high2']*np.pi,params['params_mock_cat.maker_params.surveys.los2'])
            ra_deg2=phi2*180.0/np.pi

            p_low2= (np.sin(np.radians(params['params_mock_cat.maker_params.surveys.delta_low2']))+1.0)/2.0
            p_high2= (np.sin(np.radians(params['params_mock_cat.maker_params.surveys.delta_high2']))+1.0)/2.0
            p2=np.random.uniform(p_low2,p_high2,params['params_mock_cat.maker_params.surveys.los2'])

            sin_delta2=2*p2-1
            delta2=np.arcsin(sin_delta2)
            dec_deg2=np.degrees(delta2)

            c2 = SkyCoord(ra=ra_deg2*u.degree, dec=dec_deg2*u.degree, frame='icrs')
            l2=c2.galactic.l.value
            b2=c2.galactic.b.value
            survey_indices2= hp.ang2pix(params['params_inference.nside'], l2, b2, lonlat=True)
            all_indices[survey_indices2]=2
            catalog[params['params_mock_cat.maker_params.surveys.los']:]=params['params_mock_cat.maker_params.surveys.name2']

        survey_indices_field=ift.Field(s_space_domain, all_indices)

        l=np.concatenate([l1,l2])
        b=np.concatenate([b1,b2])
        ra=np.concatenate([ra_deg1,ra_deg2])
        dec=np.concatenate([dec_deg1,dec_deg2])

        plot = ift.Plot()
        plot.add(survey_indices_field)
        plot.output(name='survey.png')



        
        cols=fits.ColDefs([
            fits.Column(name='ra', format='D', array=ra, unit=''),
            fits.Column(name='dec', format='D', array=dec, unit=''),
            fits.Column(name='l', format='D', array=l, unit='deg'),
            fits.Column(name='b', format='D', array=b, unit='deg'),
            fits.Column(name='pos_err', format='D', array=pos_err, unit='deg'),
            fits.Column(name='rm', format='D', array=rm, unit='rad m-2'),
            fits.Column(name='rm_err', format='D', array=rm_err, unit='rad m-2'),
            fits.Column(name='rm_width', format='D', array=rm_width, unit='rad m-2'),
            fits.Column(name='rm_width_err', format='D', array=rm_width_err, unit='rad m-2'),
            fits.Column(name='complex_flag', format='2A', array=complex_flag, unit=''),
            fits.Column(name='complex_test', format='80A', array=complex_test, unit=''),
            fits.Column(name='rm_method', format='40A', array=rm_method, unit=''),
            fits.Column(name='ionosphere', format='40A', array=ionosphere, unit=''),
            fits.Column(name='Ncomp', format='K', array=Ncomp, unit=''),
            fits.Column(name='stokesI', format='D', array=stokesI, unit='Jy'),
            fits.Column(name='stokesI_err', format='D', array=stokesI_err, unit='Jy'),
            fits.Column(name='spectral_index', format='D', array=spectral_index, unit=''),
            fits.Column(name='spectral_index_err', format='D', array=spectral_index_err, unit=''),
            fits.Column(name='reffreq_I', format='D', array=reffreq_I, unit='Hz'),
            fits.Column(name='polint', format='D', array=polint, unit='Jy'),
            fits.Column(name='polint_err', format='D', array=polint_err, unit='Jy'),
            fits.Column(name='pol_bias', format='50A', array=pol_bias, unit=''),
            fits.Column(name='flux_type', format='50A', array=flux_type, unit=''),
            fits.Column(name='aperture', format='D', array=aperture, unit=''),
            fits.Column(name='fracpol', format='D', array=fracpol, unit=''),
            fits.Column(name='fracpol_err', format='D', array=fracpol_err, unit=''),
            fits.Column(name='polangle', format='D', array=polangle, unit='deg'),
            fits.Column(name='polangle_err', format='D', array=polangle_err, unit='deg'),
            fits.Column(name='reffreq_pol', format='D', array=reffreq_pol, unit='Hz'),
            fits.Column(name='stokesQ', format='D', array=stokesQ, unit='Jy'),
            fits.Column(name='stokesQ_err', format='D', array=stokesQ_err, unit='Jy'),
            fits.Column(name='stokesU', format='D', array=stokesU, unit='Jy'),
            fits.Column(name='stokesU_err', format='D', array=stokesU_err, unit='Jy'),
            fits.Column(name='derot_polangle', format='D', array=derot_polangle, unit='deg'),
            fits.Column(name='derot_polangle_err', format='D', array=derot_polangle_err, unit='deg'),
            fits.Column(name='stokesV', format='D', array=stokesV, unit='Jy'),
            fits.Column(name='stokesV_err', format='D', array=stokesV_err, unit='Jy'),
            fits.Column(name='beam_maj', format='D', array=beam_maj, unit='deg'),
            fits.Column(name='beam_min', format='D', array=beam_min, unit='deg'),
            fits.Column(name='beam_pa', format='D', array=beam_pa, unit='deg'),
            fits.Column(name='reffreq_beam', format='D', array=reffreq_beam, unit='Hz'),
            fits.Column(name='minfreq', format='D', array=minfreq, unit='Hz'),
            fits.Column(name='maxfreq', format='D', array=maxfreq, unit='Hz'),
            fits.Column(name='channelwidth', format='D', array=channelwidth, unit='Hz'),
            fits.Column(name='Nchan', format='K', array=Nchan, unit=''),
            fits.Column(name='rmsf_fwhm', format='D', array=rmsf_fwhm, unit='rad m-2'),
            fits.Column(name='noise_chan', format='D', array=noise_chan, unit='Jy beam-1'),
            fits.Column(name='telescope', format='80A', array=telescope, unit=''),
            fits.Column(name='int_time', format='D', array=int_time, unit='s'),
            fits.Column(name='epoch', format='D', array=epoch, unit=''),
            fits.Column(name='interval', format='D', array=interval, unit='d'),
            fits.Column(name='leakage', format='D', array=leakage, unit=''),
            fits.Column(name='beamdist', format='D', array=beamdist, unit='deg'),
            fits.Column(name='catalog', format='50A', array=catalog, unit=''),
            fits.Column(name='dataref', format='400A', array=dataref, unit=''),
            fits.Column(name='cat_id', format='40A', array=cat_id, unit=''),
            fits.Column(name='type', format='40A', array=types, unit=''),
            fits.Column(name='notes', format='200A', array=notes, unit=''),
            fits.Column(name='z_best', format='D', array=z_best, unit='')])


        hdu=fits.BinTableHDU.from_columns(cols)
        survey_cat_path=params['params_inference.cat_path']+params['params_mock_cat.maker_params.surveys.name']+'_catalog.fits'
        hdu.writeto(survey_cat_path, overwrite=True)
        cat = read_FITS(survey_cat_path)
        quantities = ['l', 'b', 'rm', 'rm_err', 'catalog', 'z_best', 'stokesI', 'type']
        data = {q: cat[q] for q in quantities}
        theta_gal, phi_gal = gal2gal(data['l'], data['b']) # converting to colatitude and logitude in radians
        data.update({'theta': theta_gal, 'phi': phi_gal})

        
        return data
    
