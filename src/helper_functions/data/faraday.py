import numpy as np
import healpy as hp
import astropy.io.fits as af
import math
import rmtable as rmt
from Functions.logger import logger, Format
from Functions.misc import equ2gal, gal2gal


def faraday(fits, old, estimated_stddev, load_additional, filter_pulsars, filter_cgps, version):
    logger.info(
        "\n" + Format.underline + "DATA LIBRARY:" + Format.end +
        " _faraday: params : \'fits\' {}, \'old\' {}, \'estimated_stddev\' {}, "
        "\'load_additional\' {}, \'filter_pulsars\' {}, \'filter_cgps\' {}, \'version\' {}"
        .format(fits, old, estimated_stddev, load_additional, filter_pulsars, filter_cgps, version))
    if not old:
        theta, phi, rm, rm_stddev, names, lengths = load_faraday_new_master(fits=fits,
                                                                            load_additional=load_additional,
                                                                            filter_pulsar=filter_pulsars,
                                                                            filter_cgps=filter_cgps, version=version)
    else:
        theta, phi, rm, rm_stddev, names, lengths = load_faraday_niels()
    if estimated_stddev:
        import pickle
        with open('./data/faraday_stddev_estimated.pickle', 'rb') as f:
            rm_stddev = pickle.load(f)
    return {'faraday_data': rm, 'faraday_stddev': rm_stddev, 'faraday_angles': [theta, phi]}


def load_faraday_new_master(fits, load_additional, filter_cgps, filter_pulsar, version):
    if fits:
        cat = rmt.read_FITS('./data/Faraday/catalog_versions/master_catalog_ver' + version + '.fits')
    else:
        cat = rmt.read_tsv('./data/Faraday/catalog_versions/master_catalog_ver' + version + '.tsv')
    logger.info("DATA LOADING: load_faraday_new_master: Loading master catalog, "
                "number of data points: {}".format(str(len(cat['rm']))))
    logger.info('DATA LOADING: load_faraday_new_master: Unique papers: '.format(len(np.unique(cat['catalog']))))
    if load_additional:
        logger.info('DATA LOADING: load_faraday_new_master: Adding additional catalogs (fits)')
        catlist = ['CGPS_table_AO.fits']
        for add_cat in catlist:
            ac = rmt.read_FITS('./data/Faraday/additional/' + add_cat)
            logger.info('DATA LOADING: load_faraday_new_master: Adding catalog {}: {} '
                        'data points added'.format(add_cat, len(ac['rm'])))
            cat.append_to_table(ac, join_type='outer')
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
    if load_additional:
        import sys
        logger.info('DATA LOADING: load_faraday_new_master: Adding additional catalogs (sys)')
        for function in ['johnston_hollitt', 'LoTSS']:
            theta_add, phi_add, rm_add, sigma_add = getattr(sys.modules[__name__], 'get_' + function)()
            sigma = np.append(sigma, sigma_add)
            rm = np.append(rm, rm_add)
            theta_gal = np.append(theta_gal, theta_add)
            phi_gal = np.append(phi_gal, phi_add)
            logger.info('DATA LOADING: load_faraday_new_master: Adding catalog {}: '
                        '{} data points added'.format(function, len(rm_add)))
            logger.info('DATA LOADING: load_faraday_new_master: New number of data points: {}'.format(len(rm)))
    logger.info('DATA LOADING: load_faraday_new_master: {} nan or zero valued sigma values, '
                'corrected to 0.5 * abs(RM)'.format(len(sigma[np.isnan(sigma)]) + len(sigma[sigma == 0])))
    sigma[np.isnan(sigma)] = 0.5 * abs(rm[np.isnan(sigma)])
    sigma[sigma == 0] = 0.5 * abs(rm[sigma == 0])
    logger.info('DATA LOADING: load_faraday_new_master: Final number of data points: {}\n'.format(len(rm)))
    return theta_gal, phi_gal, rm, sigma, cat['catalog'], None


def load_faraday_niels(public=False):
    import h5py
    file = h5py.File('./data/oppermann/samples_subset.hdf5')
    lat_gal = file['sourceinfo/Gal_lat'][:]
    lon_gal = file['sourceinfo/Gal_lon'][:]
    theta_gal, phi_gal = gal2gal(lon_gal, lat_gal)
    RM = file['sourceinfo/observed'][:]
    sigma = file['sourceinfo/sigma_observed'][:]
    names_from_file = file['sourceinfo/catalog'][:]
    names = []
    lengths = []
    for cc in range(len(names_from_file)):
        if names_from_file[cc] not in names:

            if not(len(names) == 0):
                lengths.append(counter)
            names.append(names_from_file[cc])
            counter = 0
        counter += 1

    lengths.append(counter)
    if sum(lengths) != len(names_from_file):
        raise ValueError('Somethings fishy')

    if public:
        return theta_gal, phi_gal, RM, sigma, names_from_file, None

    hammond = get_hammond()
    theta_gal[np.where(names_from_file == b'Hammond')] = hammond[0]
    phi_gal[np.where(names_from_file == b'Hammond')] = hammond[1]
    RM[np.where(names_from_file == b'Hammond')] = hammond[2]
    sigma[np.where(names_from_file == b'Hammond')] = hammond[3]

    osullivan = get_osullivan()
    theta_gal[np.where(names_from_file == b"O'Sullivan")] = osullivan[0]
    phi_gal[np.where(names_from_file == b"O'Sullivan")] = osullivan[1]
    RM[np.where(names_from_file == b"O'Sullivan")] = osullivan[2]
    sigma[np.where(names_from_file == b"O'Sullivan")] = osullivan[3]

    schnitzeler = get_schnitzeler()
    theta_gal[np.where(names_from_file == b'Schnitzeler')] = schnitzeler[0]
    phi_gal[np.where(names_from_file == b'Schnitzeler')] = schnitzeler[1]
    RM[np.where(names_from_file == b'Schnitzeler')] = schnitzeler[2]
    sigma[np.where(names_from_file == b'Schnitzeler')] = schnitzeler[3]

    johnston_hollitt = get_johnston_hollitt()
    theta_gal[np.where(names_from_file == b'Johnston-Hollitt A')] = johnston_hollitt[0]
    phi_gal[np.where(names_from_file == b'Johnston-Hollitt A')] = johnston_hollitt[1]
    RM[np.where(names_from_file == b'Johnston-Hollitt A')] = johnston_hollitt[2]
    sigma[np.where(names_from_file == b'Johnston-Hollitt A')] = johnston_hollitt[3]

    mao_lmc = get_mao_lmc()
    theta_gal[np.where(names_from_file == b'Mao LMC')] = mao_lmc[0]
    phi_gal[np.where(names_from_file == b'Mao LMC')] = mao_lmc[1]
    RM[np.where(names_from_file == b'Mao LMC')] = mao_lmc[2]
    sigma[np.where(names_from_file == b'Mao LMC')] = mao_lmc[3]

    # sigma[np.where(sigma == 0)] = RM[np.where(sigma == 0)]
    sigma += 0.5 * (sigma == 0.)

#    sigma[np.where(sigma == 0)] = RM[np.where(sigma == 0)]
    sigma += 0.5 * (sigma == 0.)
    return theta_gal, phi_gal, RM, sigma, names, lengths


def get_LoTSS(celestial=False):
    with af.open('./data/Faraday/additional/LOFAR_RMgrid_forSebastian.fits') as f:
        rm = f[1].data['RM']
        rm_e = f[1].data['RMerror']
        ra = f[1].data['ra']
        dec = f[1].data['dec']
    r = hp.Rotator(coord=['C', 'G'], deg=False)
    colat_equ_rad = (90. - dec)*np.pi/180.
    lon_equ_rad = ra/360 * 2. * np.pi
    if celestial:
        return colat_equ_rad, lon_equ_rad, rm, rm_e
    return *r(colat_equ_rad, lon_equ_rad), rm, rm_e


def get_hammond():
    data = np.array(af.getdata('./data/oppermann/all/hammond.fits'))
    ra_h = np.array([data[i][2] for i in range(len(data))])
    ra_m = np.array([data[i][3] for i in range(len(data))])
    ra_s = np.array([data[i][4] for i in range(len(data))])
    dec_d = np.array([-data[i][6] for i in range(len(data))])
    sign = np.array([math.copysign(1.,dec_d[i]) for i in range(len(data))])
    dec_m = np.array([data[i][7] for i in range(len(data))])*sign
    dec_s = np.array([data[i][8] for i in range(len(data))])*sign
    theta_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[0]
    phi_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[1]
    rm = np.array([data[i][15] for i in range(len(data))])
    rm_e = np.array([data[i][16] for i in range(len(data))])
    return theta_gal, phi_gal, rm, rm_e


def get_osullivan():
    data = np.genfromtxt('./data/oppermann/all/osullivan.dat')
    ra_h = np.array([data[i, 0] for i in range(len(data))])
    ra_m = np.array([data[i, 1] for i in range(len(data))])
    ra_s = np.array([data[i, 2] for i in range(len(data))])
    dec_d = np.array([data[i, 4] for i in range(len(data))])
    sign = np.array([math.copysign(1., dec_d[i]) for i in range(len(data))])
    dec_m = np.array([data[i, 5] for i in range(len(data))])*sign
    dec_s = np.array([data[i, 6] for i in range(len(data))])*sign
    theta_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[0]
    phi_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[1]
    rm = np.array([data[i, 10] for i in range(len(data))])
    rm_e = np.array([data[i, 11] for i in range(len(data))])
    return theta_gal, phi_gal, rm, rm_e


def get_schnitzeler():
    data = np.genfromtxt('./data/oppermann/all/schnitzeler.dat')
    ra_h = np.array([data[i, 0] for i in range(len(data))])
    ra_m = np.array([data[i, 1] for i in range(len(data))])
    ra_s = np.array([data[i, 2] for i in range(len(data))])
    dec_d = np.array([data[i, 3] for i in range(len(data))])
    sign = np.array([math.copysign(1., dec_d[i]) for i in range(len(data))])
    dec_m = np.array([data[i, 4] for i in range(len(data))])*sign
    dec_s = np.array([data[i, 5] for i in range(len(data))])*sign
    theta_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[0]
    phi_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[1]
    rm = np.array([data[i, 14] for i in range(len(data))])
    rm_e = np.array([data[i, 15] for i in range(len(data))])
    return theta_gal, phi_gal, rm, rm_e


def get_johnston_hollitt():
    data = np.genfromtxt('./data/oppermann/all/johnston-hollitt-new.dat', encoding='utf-8')
    ra_h = np.array([data[i, 0] for i in range(len(data))])
    ra_m = np.array([data[i, 1] for i in range(len(data))])
    ra_s = np.array([data[i, 2] for i in range(len(data))])
    dec_d = np.array([data[i, 4] for i in range(len(data))])
    sign = np.array([math.copysign(1., dec_d[i]) for i in range(len(data))])
    dec_m = np.array([data[i, 5] for i in range(len(data))]) * sign
    dec_s = np.array([data[i, 6] for i in range(len(data))]) * sign
    theta_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[0]
    phi_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[1]
    rm = np.array([data[i, 12] for i in range(len(data))])
    rm_e = np.array([data[i, 13] for i in range(len(data))])
    return theta_gal, phi_gal, rm, rm_e


def get_johnston_hollitt_b():
    theta_gal, phi_gal, RM, sigma, names, lengths = load_faraday_niels(public=True)
    return theta_gal[np.where(names == b'Johnston-Hollitt B')], phi_gal[np.where(names == b'Johnston-Hollitt B'), ], \
           RM[np.where(names == b'Johnston-Hollitt B'), ], sigma[np.where(names == b'Johnston-Hollitt B'), ]


def get_mao_lmc():
    data = np.genfromtxt('./data/oppermann/all/mao_lmc.dat', encoding='utf-8')
    ra_h = np.array([data[i, 0] for i in range(len(data))])
    ra_m = np.array([data[i, 1] for i in range(len(data))])
    ra_s = np.array([data[i, 2] for i in range(len(data))])
    dec_d = np.array([data[i, 4] for i in range(len(data))])
    sign = np.array([math.copysign(1., dec_d[i]) for i in range(len(data))])
    dec_m = np.array([data[i, 5] for i in range(len(data))]) * sign
    dec_s = np.array([data[i, 6] for i in range(len(data))]) * sign
    theta_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[0]
    phi_gal = equ2gal(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s)[1]
    rm = np.array([data[i, 12] for i in range(len(data))])
    rm_e = np.array([data[i, 13] for i in range(len(data))])
    return theta_gal, phi_gal, rm, rm_e


def get_lofar():
    data = np.genfromtxt('./data/Faraday/additional/RMpairs_Table_2.dat', encoding='utf-8', dtype=str)
    ra = data[:, 1]
    ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = ([] for i in range(6))
    for r in ra:
        ra_h.append(float(r[0:2]))
        ra_m.append(float(r[3:5]))
        ra_s.append(float(r[6:]))
    dec = data[:, 2]
    for d in dec:
        dec_d.append(float(d[0:3]))
        dec_m.append(float(d[4:6])*np.sign(float(d[0:2])))
        dec_s.append(float(d[7:])*np.sign(float(d[0:2])))
    theta_gal, phi_gal = equ2gal(np.asarray(ra_h), np.asarray(ra_m), np.asarray(ra_s),
                                 np.asarray(dec_d), np.asarray(dec_m), np.asarray(dec_s))
    rm = data[:, 4].astype(float)
    rm_error = data[:, 5].astype(float)

    return theta_gal, phi_gal, rm, rm_error


def get_costa():
    data = np.genfromtxt('./data/Faraday/additional/costa_table.dat', encoding='utf-8', dtype=str)
    ra = data[:, 1]
    ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = ([] for i in range(6))
    for r in ra:
        ra_h.append(float(r[0:2]))
        ra_m.append(float(r[3:5]))
        ra_s.append(float(r[6:]))
    dec = data[:, 2]
    for d in dec:
        dec_d.append(float(d[0:3]))
        dec_m.append(float(d[4:6])*np.sign(float(d[0:2])))
        dec_s.append(float(d[7:])*np.sign(float(d[0:2])))

    theta_gal, phi_gal = equ2gal(np.asarray(ra_h), np.asarray(ra_m), np.asarray(ra_s),
                                 np.asarray(dec_d), np.asarray(dec_m), np.asarray(dec_s))
    rm = data[:, 3].astype(float)
    rm_error = data[:, 4].astype(float)

    return theta_gal, phi_gal, rm, rm_error

