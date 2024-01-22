import nifty8 as ift
import numpy as np
import healpy as hp


def gradient_map(ax, m, gradient_props, arrowprops):
    if isinstance(m, ift.Field):
        m = m.val_rw()
    assert hp.isnpixok(len(m)), 'input map is not a healpix map'
    nside = hp.npix2nside(len(m))
    nside_max = gradient_props['nside_max']
    gc = gradient_props['gradient corrections']
    if nside >= nside_max:
        hp.ud_grade(m, nside_max)
    x_lm = hp.map2alm(m, pol=False)
    x, dtheta, dphi = hp.alm2map_der1(x_lm, min(nside_max, nside))
    dtheta /= hp.nside2npix(min(nside_max, nside))/gc
    dphi /= hp.nside2npix(min(nside_max, nside))/gc

    print('gradients', dtheta.max(), dphi.max())

    theta_start, phi_start = hp.pix2ang(min(nside_max, nside), np.arange(hp.nside2npix(min(nside_max, nside))))
    theta_end = theta_start + dtheta
    phi_end = phi_start + dphi

    proj = hp.projector.MollweideProj()
    x, y = proj.ang2xy(theta_end, phi_end, lonlat=False)

    x_end, y_end = proj.ang2xy(theta_start, phi_start, lonlat=False)
    r = np.sqrt(((x_end - x) + (y_end - y))**2)
    x[r > 0.5], y[r > 0.5], x_end[r > 0.5], y_end[r > 0.5] = (0, 0, 0, 0, )
    for xs, ys, xe, ye in zip(x, y, x_end, y_end):
        ax.annotate('', xy=(xs, ys, ), xytext=(xe, ye, ),
                    arrowprops=arrowprops)


def equ2gal(ra_h, ra_m, ra_s, dec_deg, dec_min, dec_sec):
    """Conversion from equatorial coordinates to galactic coordinates.
    Note that while lon_equ is in [0, 2pi] due to the ra definition, lon gal is in in [-pi, pi].
    The two conventions lead to numerically equivalent results in all further processing."""
    r = hp.Rotator(coord=['C', 'G'], deg=False)
    colat_equ_rad = (90.-dec_deg-dec_min/60.-dec_sec/3600.)*np.pi/180.
    lon_equ_rad = (ra_h/24.+ra_m/1440.+ra_s/86400.)*2.*np.pi
    colat_gal_rad, lon_gal_rad = r(colat_equ_rad, lon_equ_rad)
    return colat_gal_rad, lon_gal_rad


def gal2gal(lon_deg, lat_deg):
    """Conversion from longitude and latitude in degrees to colatitude and longitude in radians
    (matching healpy convention)."""
    colat_rad = (90.-lat_deg)*np.pi/180.
    lon_rad = lon_deg*np.pi/180.
    return colat_rad, lon_rad


def density_estimation(m1, m2, xmin, xmax, ymin, ymax, nbins):
    x, y = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([m1, m2])
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    return x, y, z


def load_fields(name, domain, a_dict):
    fluc = {'prefix': name + '_'}
    fluc.update(a_dict['fluctuations'])
    cfmaker = ift.CorrelatedFieldMaker(name + '_')
    cfmaker.set_amplitude_total_offset(**a_dict['offset'])

    cfmaker.add_fluctuations(domain[0], **fluc)
    return cfmaker.finalize(), cfmaker.amplitude


def explicit_jacobian_consistency(op, loc, tol=1e-8, ntries=100):
    """
    Checks the Jacobian of an operator against its finite difference
    approximation.

    Computes the Jacobian with finite differences and compares it to the
    implemented Jacobian.

    Parameters
    ----------
    op : Operator
        Operator which shall be checked.
    loc : Field or MultiField
        An Field or MultiField instance which has the same domain
        as op. The location at which the gradient is checked
    tol : float
        Tolerance for the check.
    """
    for _ in range(ntries):
        lin = op(ift.Linearization.make_var(loc))
        loc2, lin2 = ift.extra._get_acceptable_location(op, loc, lin)
        dir = loc2-loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(ift.Linearization.make_var(locmid))
            dirder = linmid.jac(dir)
            numgrad = (lin2.val-lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            if i % 10 == 0:
                # ratio = dirder.val/numgrad.val
                print('xtol: ', xtol)
                print('dirder: ', dirder.val)
                print('numgrad: ', numgrad.val)
                # print('ratio: ', ratio)
                # print('where: ', ratio[abs(numgrad.val-dirder.val) > xtol], numgrad.val[abs(numgrad.val-dirder.val) > xtol], dirder.val[abs(numgrad.val-dirder.val) > xtol])
                print('val: ', op(loc).val)
            if (abs(numgrad-dirder) <= xtol).all():
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext


def CalculateTheGalacticEnd(r_0, r_c, z_c, nside):
    import healpy as hp
    chi_v = np.zeros(hp.nside2npix(nside))
    dx = 1
    r = np.arange(50000)
    for i in [r_0, r_c, z_c]:
        i /= dx
    for i in range(0, hp.nside2npix(nside)):
        phi, theta = hp.pixelfunc.pix2ang(nside, i, nest=False, lonlat=False)
        r_z = np.sqrt((r*np.sin(phi)*np.cos(theta) - r_0)**2 + (r*np.sin(phi)*np.sin(theta))**2)
        z_z = r*np.cos(phi)
        chi_v[i] = np.trapz(np.exp(-abs(r_z/r_c))*np.cosh(z_z/z_c)**(-2))
        if i % 10000 == 0:
            print(i)
    return chi_v
