import os
import healpy as hp
import numpy as np
try: 
    import pygedm
    has_pygedm = True
except ImportError:
    has_pygedm = False
from ..logger import logger, Format



def load_ymw_sky(model, mode, params):

    filename = params['params_inference.dm_path'] + '{}_allsky_{}_{}.fits'.format(model, params['params_inference.nside'], mode)
    if os.path.exists(filename):
        logger.info("DATA LOADING: load_ymw: loading existing file {} with nside {}".format(filename, params['params_inference.nside']))
        d_2016 = hp.read_map(filename)
    elif has_pygedm:
        logger.info("DATA LOADING: load_ymw: Generating fullsky healpix dm map based on ymw model... (this may take a while)")
        pix_id = np.arange(hp.nside2npix(params['params_inference.nside']))
        gl, gb = hp.pix2ang(params['params_inference.nside'], pix_id, lonlat=True)
        d_2016 = np.zeros_like(pix_id, dtype='float32')

        for ii in pix_id:
            dm, tau = pygedm.dist_to_dm(gl[ii], gb[ii], 100000, mode=mode, method=model)
            if ii % 100000 == 0:
                logger.info("DATA LOADING: load_yml: converting pixel #{} of {}".format(ii, 12 * params['params_inference.nside'] ** 2))
            d_2016[ii] = dm.value
        logger.info("DATA LOADING: load_yml: writing to file {} with nside {}\n".format(filename, params['params_inference.nside']))
        hp.write_map(filename, d_2016, coord='G')
    else:
        raise ImportError("YMW file not found on disk and pygedm not installed")
    return d_2016
