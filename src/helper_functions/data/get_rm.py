import numpy as np
import libs as Egf

from .rmtable import read_FITS
from ..logger import logger, Format
from ..misc import gal2gal


def get_rm(version, filter_pulsars, default_error_level):
    logger.info(
        "\n" + Format.underline + "DATA LIBRARY:" + Format.end +
        " _faraday: params : \'filter_pulsars\' {}, \'default_error_level\' {}, \'version\' {}"
        .format(filter_pulsars, default_error_level, version))

    cat = read_FITS(Egf.config['file_params']['fits_file_path'] + version + Egf.config['file_params']['fits_ext'])

    logger.info("DATA LOADING: load_faraday_new_master: Loading master catalog, "
                "number of data points: {}".format(str(len(cat['rm']))))
    logger.info('DATA LOADING: load_faraday_new_master: Unique papers: '.format(len(np.unique(cat['catalog']))))
    logger.info('DATA LOADING: load_faraday_new_master: New number of data points: {}'.format(len(cat['rm'])))
    if filter_pulsars:
        logger.info('DATA LOADING: load_faraday_new_master: Filtering Pulsars:'
                    ' {} filtered'.format(len(cat[(cat['type'] == 'Pulsar')])))
        cat = cat[~(cat['type'] == 'Pulsar')]
        logger.info('DATA LOADING: load_faraday_new_master: New number of data points: {}'.format(len(cat['rm'])))

    if version == "custom":
        quantities = ['l', 'b', 'rm', 'rm_err', 'catalog', 'z_best', 'stokesI']
    else:
        quantities = ['l', 'b', 'rm', 'rm_err', 'catalog']

    data = {q: cat[q] for q in quantities}

    data['rm_err'][data['catalog'] == '2009ApJ...702.1230T'] *= 1.22  # Taylor et al error correction, see
    theta_gal, phi_gal = gal2gal(data['l'], data['b']) # converting to colatitude and logitude in radians
    data.update({'theta': theta_gal, 'phi': phi_gal})
    faulty_sigmas = np.isnan(data['rm_err']) | (data['rm_err'] == 0)
    data['rm_err'][np.isnan(data['rm_err'])] = 0.5 * abs(data['rm'][np.isnan(data['rm_err'])])
    data['rm_err'][data['rm_err'] == 0] = 0.5 * abs(data['rm'][data['rm_err'] == 0])
    logger.info('DATA LOADING: load_faraday_new_master: {} nan or zero valued sigma values, '
                'corrected to 0.5 * abs(RM)'.format(sum(faulty_sigmas)))
    #data['rm_err'][faulty_sigmas] = default_error_level * abs(data['rm'][faulty_sigmas])
    logger.info('DATA LOADING: load_faraday_new_master: Final number of data points: {}\n'.format(len(data['rm'])))
    return data
