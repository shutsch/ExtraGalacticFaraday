import os
import numpy as np
import nifty7 as ift
from .nifty_cmaps import ncmap
import matplotlib.pyplot as pl
import matplotlib.colors as cm


def energy_plotting(array_dict, path):
    path += 'energy/'
    if not os.path.exists(path):
        os.makedirs(path)
    for key, e in array_dict.items():
        pl.figure()
        pl.plot(np.arange(len(e)), e, label=key + '_log_energy_iteration_' + str(len(e) - 1))
        pl.legend()
        pl.yscale('log')
        pl.savefig(path + key + '_log_energy.png')
        pl.close()
    pl.figure()
    pl.yscale('log')
    pl.legend()
    pl.savefig(path + 'all_log_energy.png')
    pl.close()


def prior_plotting(samples, amplitudes, sky_fields, hist, path, **kwargs):
    """

    :param hist:
    :param samples:
    :param amplitudes:
    :param sky_fields:
    :param path:
    :return:
    """
    path += 'prior/'
    if not os.path.exists(path):
        os.makedirs(path)
    power_plotting(amplitudes, sky_fields, samples, 'prior', path)
    hpath = path + 'hist/'
    if not os.path.exists(hpath):
        os.makedirs(hpath)
    for key, hfield in hist.items():
        for i, sample in enumerate(samples):
            hval = hfield.force(sample).val
            if key in kwargs:
                hmax = kwargs[key].get('max', None)
                hmin = kwargs[key].get('min', None)
                bins = kwargs[key].get('bins', 100)
                hmax = [hmax, ] if not isinstance(hmax, list) else hmax
                hmin = [hmin, ] if not isinstance(hmin, list) else hmin
                bins = [bins, ] if not isinstance(bins, list) else bins
                lmax = max(len(hmax), len(hmin), len(bins))
                if lmax > 1:
                    hmax = lmax * hmax if len(hmax) == 1 else hmax
                    hmin = lmax * hmin if len(hmin) == 1 else hmin
                    bins = lmax * bins if len(bins) == 1 else bins
            else:
                hmax, hmin, bins = [None, ], [None, ], [100, ]
            for i, (hmi, hma, bi,) in enumerate(zip(hmin, hmax, bins)):
                pl.figure()
                if (hmi is None) & (hma is None):
                    pl.hist(hval, bins=bi)
                else:
                    hmi = hval.min() if hmi is None else hmi
                    hma = hval.max() if hma is None else hma
                    rang = (hmi, hma,)
                    pl.hist(hval, range=rang, bins=bi)

                pl.savefig(hpath + key + str(i) + '.png')
                pl.close()

    spath = path + 'sky/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    for sky in sky_fields:

        plot = ift.Plot()
        sc = ift.StatCalculator()
        for sample in samples:
            sc.add(sample)
            m = sky_fields[sky].force(sample)
            cmap = None
            if sky in kwargs:
                if 'cmap' in kwargs[sky]:
                    cmap = getattr(cm, kwargs[sky]['cmap'])
            plot.add(m, cmap=cmap)
        plot.add(sky_fields[sky].force(sc.mean), title='mean')
        plot.output(name=spath + sky + "_prior.png")


def data_plotting(data_dict, path, **kwargs):
    path += 'data/'
    if not os.path.exists(path):
        os.makedirs(path)
    for data in data_dict:
        if data[:9] == 'projected':
            plot = ift.Plot()
            if data in kwargs:
                for zmax, zmin in zip(kwargs[data]['max'], kwargs[data]['min']):
                    plot.add(data_dict[data], title=data, vmax=zmax, vmin=zmin)
            else:
                plot.add(data_dict[data], title=data)
            ny = len(plot._plots)
            bd = plot_pd.get(data, {'t': None})
            if 'color' in bd:
                if 'cmap' in kwargs[sky]:
                    cmap = getattr(cm, kwargs[sky]['cmap'])
            else:
                cmap = None
            plot.output(nx=1, ny=ny, xsize=12, ysize=ny * 12, cmap=cmap, name=path + data + ".png")
        else:
            dval = data_dict[data]
            if isinstance(dval, ift.Field):
                dval = data_dict[data].val
            if data in plot_pd:
                hmax = plot_pd[data].get('max', None)
                hmin = plot_pd[data].get('min', None)
                bins = plot_pd[data].get('bins', 100)
                hmax = [hmax, ] if not isinstance(hmax, list) else hmax
                hmin = [hmin, ] if not isinstance(hmin, list) else hmin
                bins = [bins, ] if not isinstance(bins, list) else bins
                lmax = max(len(hmax), len(hmin), len(bins))
                if lmax > 1:
                    hmax = lmax*hmax if len(hmax) == 1 else hmax
                    hmin = lmax*hmin if len(hmin) == 1 else hmin
                    bins = lmax*bins if len(bins) == 1 else bins
            else:
                hmax, hmin, bins = [None, ], [None, ], [100, ]
            for i, (hmi, hma, bi,) in enumerate(zip(hmin, hmax, bins)):
                pl.figure()
                if (hmi is None) & (hma is None):
                    pl.hist(dval, bins=bi)
                else:
                    hmi = dval.min() if hmi is None else hmi
                    hma = dval.max() if hma is None else hma
                    rang = (hmi, hma,)
                    pl.hist(dval, range=rang, bins=bi)
                pl.savefig(path + data + '_hist_' + str(i) + '.png')
                pl.close()


def power_plotting(model, samples, name, path, from_power_model, string=None):
    amp_path = path + 'power/' + name + '/'
    if not os.path.exists(amp_path):
        os.makedirs(amp_path)
    if from_power_model:
        amp_model_samples = [model.force(s) for s in samples]
        plo = ift.Plot()
        linewidth = [1.] * len(amp_model_samples) + [3., ]
        alpha = [.5] * len(amp_model_samples) + [1., ]
        plo.add(amp_model_samples, title="Sampled Posterior Power Spectrum, " + name, linewidth=linewidth, alpha=alpha)
        plo.output(name=amp_path + name + '_' + string + ".png")

    else:
        plo = ift.Plot()
        ht = ift.HarmonicTransformOperator(model.target[0].get_default_codomain(), model.target[0])
        plo.add(
            [ift.power_analyze(ht.adjoint(model.force(s))) for s in samples],
            title="Calculated Power Spectrum, " + name)
        plo.output(name=amp_path + name + string + ".png")


def sky_map_plotting(model, plot_obj, name, path, string=None, **kwargs):
    sky_path = path + 'sky/' + name + '/'
    if not os.path.exists(sky_path):
        os.makedirs(sky_path)
    plot = ift.Plot()
    if isinstance(plot_obj, list):
        sc = ift.StatCalculator()
        for sample in plot_obj:
            print(name, type(model), type(sample), sample)
            sc.add(model.force(sample))
        m = sc.mean
    else:
        m = model.force(plot_obj)
    if 'cmap' in kwargs:
        try:
            kwargs['cmap'] = getattr(ncmap, kwargs['cmap'])
        except AttributeError:
            kwargs['cmap'] = getattr(cm, kwargs['cmap'])

    plot.add(m, title="mean", **kwargs)
    if len(plot_obj) > 1:
        if 'cmap_stddev' in kwargs:
            try:
                kwargs['cmap'] = getattr(ncmap, kwargs['cmap_stddev'])
            except AttributeError:
                kwargs['cmap'] = getattr(cm, kwargs['cmap_stddev'])
        plot.add(ift.sqrt(sc.var), **kwargs)
    if len(plot_obj) == 1:
        nx = 1
        ny = len(plot._plots)
    else:
        nx = 2
        ny = int(len(plot._plots) / 2)
    plot.output(nx=nx, ny=ny, xsize=2 * 12, ysize=ny * 12, name=sky_path + name + '_' + string + ".png")
