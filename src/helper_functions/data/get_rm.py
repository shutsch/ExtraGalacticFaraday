import numpy as np

from .rmtable import read_FITS
from ..logger import logger, Format
from ..misc import gal2gal


def get_rm(filter_pulsars, filter_cgps, version):
    logger.info(
        "\n" + Format.underline + "DATA LIBRARY:" + Format.end +
        " _faraday: params : \'filter_pulsars\' {}, \'filter_cgps\' {}, \'version\' {}"
        .format(filter_pulsars, filter_cgps, version))
    theta, phi, rm, rm_stddev, = load_faraday_rm_table(filter_pulsar=filter_pulsars,
                                                       filter_cgps=filter_cgps, version=version)
    return {'faraday_data': rm, 'faraday_stddev': rm_stddev, 'faraday_angles': [theta, phi]}


def load_faraday_rm_table(filter_cgps, filter_pulsar, version):
    cat = read_FITS('./data/Faraday/catalog_versions/master_catalog_ver' + version + '.fits')
    logger.info("DATA LOADING: load_faraday_new_master: Loading master catalog, "
                "number of data points: {}".format(str(len(cat['rm']))))
    logger.info('DATA LOADING: load_faraday_new_master: Unique papers: '.format(len(np.unique(cat['catalog']))))
    logger.info('DATA LOADING: load_faraday_new_master: New number of data points: {}'.format(len(cat['rm'])))
    if filter_pulsar:
        logger.info('DATA LOADING: load_faraday_new_master: Filtering Pulsars:'
                    ' {} filtered'.format(len(cat[(cat['type'] == 'Pulsar')])))
        cat = cat[~(cat['type'] == 'Pulsar')]
        logger.info('DATA LOADING: load_faraday_new_master: New number of data points: {}'.format(len(cat['rm'])))
    l = np.asarray(cat['l'])
    b = np.asarray(cat['b'])
    sigma = np.asarray(cat['rm_err'])
    sigma[cat['catalog'] == '2009ApJ...702.1230T'] *= 1.22
    rm = np.asarray(cat['rm'])
    if filter_cgps:
        ca = np.asarray(cat['catalog'])
        l_2003 = l[ca == b'2003ApJS..145..213B']
        l_new = l[ca == b'New CGPS (Van Eck et al 2020 in prep)']
        b_2003 = b[ca == b'2003ApJS..145..213B']
        b_new = b[ca == b'New CGPS (Van Eck et al 2020 in prep)']
        rm_2003 = rm[ca == b'2003ApJS..145..213B']
        rm_new = rm[ca == b'New CGPS (Van Eck et al 2020 in prep)']
        sigma_2003 = sigma[ca == b'2003ApJS..145..213B']
        sigma_new = sigma[ca == b'New CGPS (Van Eck et al 2020 in prep)']

        l_no_cgps = l[(ca != b'2003ApJS..145..213B') & (ca != b'New CGPS (Van Eck et al 2020 in prep)')]
        b_no_cgps = b[(ca != b'2003ApJS..145..213B') & (ca != b'New CGPS (Van Eck et al 2020 in prep)')]
        rm_no_cgps = rm[(ca != b'2003ApJS..145..213B') & (ca != b'New CGPS (Van Eck et al 2020 in prep)')]
        sigma_no_cgps = sigma[(ca != b'2003ApJS..145..213B') & (ca != b'New CGPS (Van Eck et al 2020 in prep)')]

        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: Data points without both CGPS surveys:'
                    ' {}'.format(len(l_no_cgps)))
        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: Data points in master:'
                    ' {}'.format(len(l)))
        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: Data points in Brown(2003):'
                    ' {}'.format(len(l_2003)))
        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: Data points in vanEck (in prep.):'
                    ' {}'.format(len(l_new)))
        joint = []
        for i, j, r, s in zip(l_2003, b_2003, rm_2003, sigma_2003):
            if (np.round(i, 2), np.round(j, 2)) in zip(np.round(l_new, 2), np.round(b_new, 2)):
                joint.append((np.round(i, 2), np.round(j, 2),))
            else:
                if s <= 0:
                    print('found negative error!', s)
                l_no_cgps = np.append(l_no_cgps, i)
                b_no_cgps = np.append(b_no_cgps, j)
                rm_no_cgps = np.append(rm_no_cgps, r)
                sigma_no_cgps = np.append(sigma_no_cgps, s)
        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: Data points in both cgps: '
                    '{}'.format(len(joint)))
        c = 0
        for i, j, r, s in zip(l_new, b_new, rm_new, sigma_new):
            if (np.round(i, 2), np.round(j, 2)) in joint:
                c += 1
        assert c == len(joint), 'Consistency check for cgps failed'

        l_no_cgps = np.append(l_no_cgps, l_new)
        b_no_cgps = np.append(b_no_cgps, b_new)
        rm_no_cgps = np.append(rm_no_cgps, rm_new)
        sigma_no_cgps = np.append(sigma_no_cgps, sigma_new)
        assert len(l) - len(l_no_cgps) == c, 'Consistency between master and counts failed for cgps'
        l = l_no_cgps
        b = b_no_cgps
        rm = rm_no_cgps
        sigma = sigma_no_cgps
        logger.info('DATA LOADING: load_faraday_new_master: Filtering CGPS: '
                    'Data points in master after filtering: {}'.format(len(l)))

    theta_gal, phi_gal = gal2gal(l, b)
    logger.info('DATA LOADING: load_faraday_new_master: {} nan or zero valued sigma values, '
                'corrected to 0.5 * abs(RM)'.format(len(sigma[np.isnan(sigma)]) + len(sigma[sigma == 0])))
    sigma[np.isnan(sigma)] = 0.5 * abs(rm[np.isnan(sigma)])
    sigma[sigma == 0] = 0.5 * abs(rm[sigma == 0])
    logger.info('DATA LOADING: load_faraday_new_master: Final number of data points: {}\n'.format(len(rm)))
    return theta_gal, phi_gal, rm, sigma
