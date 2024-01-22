import nifty8 as ift
import numpy as np
import healpy as hp

class SkyProjector(ift.LinearOperator):
    def __init__(self, domain, target, theta, phi, abs_latitude_cut=None):
        self._domain = ift.makeDomain(domain)
        if abs_latitude_cut is not None:
            self.lat_cut_high = (abs_latitude_cut + 90) / 90 * np.pi
            self.lat_cut_low = (- abs_latitude_cut + 90) / 90 * np.pi
            indices = (theta < self.lat_cut_high) & (theta > self.lat_cut_low)
            theta = theta[indices]
            phi = phi[indices]
            self._target = ift.makeDomain(ift.UnstructuredDomain((len(theta),)))
        else:
            self._target = ift.makeDomain(target)
        self._pixels = self.calc_pixels(self._domain[0].nside, self._target[0].shape[0],
                                        theta, phi).astype(int)

        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _times(self, x):
        return ift.Field(self.target, x.val[self._pixels])

    def _adjoint_times(self, x):
        ns = self.domain[0].nside
        pmap = np.zeros(12 * ns ** 2)
        for i in range(len(self._pixels)):
            pmap[self._pixels[i]] += x.val[i]
        # pmap[self._pixels] += x.val
        return ift.Field(self.domain, pmap)

    def counts(self, zero_to_one=True):
        ns = self.domain[0].nside
        pmap = np.zeros(12 * ns ** 2)
        pmap[self._pixels] += 1
        if zero_to_one:
            pmap[pmap == 0] = 1
        return ift.Field(self.domain, pmap)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return self._times(x)
        return self._adjoint_times(x)

    @staticmethod
    def calc_pixels(nside, ndata, colat_rad, lon_rad):
        """Calculating the pixel numbers corresponding to the data points.
        Note that longitude can be in [-2pi, 2pi]"""
        pixels = np.zeros((ndata,))
        for i in range(ndata):
            pixels[i] = hp.ang2pix(nside, colat_rad[i], lon_rad[i])
        return pixels
