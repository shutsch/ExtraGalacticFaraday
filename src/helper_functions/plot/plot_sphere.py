import os
import numpy as np
import nifty7 as ift
import nifty_cmaps as ncm
import matplotlib.pyplot as pl
import matplotlib.colors as cm


def energy_plotting(array_dict, path):
    path += 'energy/'
    if not os.path.exists(path):
        os.makedirs(path)
    for key, e in array_dict.items():
        n = len(e) - 1
        pl.figure()
        pl.plot(np.arange(len(e)), e, label=key + '_energy_iteration_' + str(n))
        pl.legend()
        pl.savefig(path + key + '_energy.png')
        pl.close()
    for key, e in array_dict.items():
        pl.figure()
        pl.plot(np.arange(len(e)), e, label=key + '_log_energy_iteration_' + str(len(e) - 1))
        pl.legend()
        pl.yscale('log')
        pl.savefig(path + key + '_log_energy.png')
        pl.close()
    pl.figure()
    for key, e in array_dict.items():
        pl.plot(np.arange(len(e)), e, label=key + '_iteration_' + str(len(e) - 1))
    pl.yscale('log')
    pl.legend()
    pl.savefig(path + 'all_log_energy')
    pl.close()


def plot_prior_samples_and_data(samples, mb, path, plot_pd):
    data_plotting(mb.data, plot_pd, path)
    prior_plotting(samples, mb, path, plot_pd)


def plot_all(string, samples, mb, path, plot_pd):
    if isinstance(samples, list):
        if len(samples) == 0:
            return
    sky_map_plotting(mb.plotting, samples, '_' + string, plot_pd, path)
    power_plotting(mb.amplitudes, mb.plotting, samples, '_' + string, path)
    plot_error(mb.noise, mb.data, string, samples, plot_pd, path)
    hist_plotting(mb.hist, '_' + string, samples, plot_pd, path)

    scatter = {}
    for key1 in mb.scatter:
        if key1 in plot_pd:
            item1 = mb.scatter[key1] if key1 in mb.scatter else mb.data[key1]

            if 'scatter_partners' in plot_pd[key1]:
                for key2 in plot_pd[key1]['scatter_partners']:
                    item2 = mb.scatter[key2] if key2 in mb.scatter else mb.data[key2]
                    if key2[-6:] == 'angles':
                        item2_theta, item2_phi = item2
                        scatter.update({key1 + ':' + key2[-6:] + '_colat': (item1, item2_theta/np.pi*180)})
                        scatter.update({key1 + ':' + key2[-6:] + '_lon_p180': (item1, (item2_phi/np.pi*180 + 180) % 360)})
                    else:
                        scatter.update({key1 + ':' + key2: (item1, item2)})
    scatter_plotting(scatter, '_' + string, samples, plot_pd, path)


def scatter_plotting(s_dict, string, samples, bounds, path):
    path += 'scatter/'

    for key, (val_1, val_2) in s_dict.items():
        try:
            scatter_path = path + key + '/'
            if not os.path.exists(scatter_path):
                os.makedirs(scatter_path)

            if isinstance(val_1, ift.Operator):
                if isinstance(samples, list):
                    sc_1 = ift.StatCalculator()
                    for sample in samples:
                        sc_1.add(val_1.force(sample))
                    val_1 = sc_1.mean
                else:
                    val_1 = val_1.force(samples)

            if isinstance(val_2, ift.Operator):
                if isinstance(samples, list):
                    sc_2 = ift.StatCalculator()
                    for sample in samples:
                        sc_2.add(val_2.force(sample))
                    val_2 = sc_2.mean
                else:
                    val_2 = val_2.force(samples)
            key_1, key_2 = key.split(':')
            val_1 = val_1.val if isinstance(val_1, ift.Field) else val_1
            val_2 = val_2.val if isinstance(val_2, ift.Field) else val_2
            #xmax, xmin = bounds.get(key_1, (val_1.max(), val_1.min(), ))
            xmax, xmin = (val_1.max(), val_1.min(),)
            ymax, ymin = (val_2.max(), val_2.min(), )

            pl.figure()
            xxx, yyy, zzz = density_estimation(val_1, val_2, xmin, xmax, ymin, ymax, 100)
            xx = np.linspace(xmin, xmax, 10)
            yy = np.linspace(ymin, ymax, 10)

            pl.contour(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, linewidths=0.9,
                       #levels=np.linspace(0.01, 1, 10)
                       )
            c1 = pl.contourf(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool,
                # levels=np.linspace(0.01, 1, 10)
                             )
            col = pl.colorbar(c1)
            pl.scatter(val_1, val_2, marker=',', s=0.5, color='black')
            pl.plot(xx, yy, '--', c='red', linewidth=0.5)
            pl.xlabel(key_1)
            pl.ylabel(key_2)
            pl.xlim([xmin, xmax, ])
            pl.ylim([ymin, ymax, ])
            pl.savefig(scatter_path + key + '_scatter_' + string + '.png', format='png', dpi=800)
            pl.close()
        except KeyError:
            continue


def hist_plotting(hist_dict, string, samples, plot_pd, path):
    path += 'hist/'
    if not os.path.exists(path):
        os.makedirs(path)
    for key, hfield in hist_dict.items():
        hpath = path + key + '/'
        if not os.path.exists(hpath):
            os.makedirs(hpath)
        if isinstance(samples, list):
            sc = ift.StatCalculator()
            for sample in samples:
                sc.add(hfield.force(sample))
            hval = sc.mean.val
        else:
            hval = hfield.force(samples).val
        if key in plot_pd:
            hmax = plot_pd[key].get('max', None)
            hmin = plot_pd[key].get('min', None)
            bins = plot_pd[key].get('bins', 100)
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

            pl.savefig(hpath + key + string + str(i) + '.png')
            pl.close()


def plot_fields(string, samples, mean, mb, path):
    path += 'test/'
    power_plotting(mb.amplitudes, {}, samples, '_' + string, path)
    field_plotting(mb.fields, mean, path)


def field_plotting(field_dict, mean, path):
    path += 'fields/'
    for field in field_dict:
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            plot = ift.Plot()
            plot.add(field_dict[field].force(mean), title=field)
            plot.output(name=path + field + ".png")
        except KeyError:
            continue


def prior_plotting(samples, mb, path, plot_pd):
    path += 'prior/'
    if not os.path.exists(path):
        os.makedirs(path)
    power_plotting(mb.amplitudes, mb.plotting, samples, 'prior', path)
    hpath = path + 'hist/'
    if not os.path.exists(hpath):
        os.makedirs(hpath)
    for key, hfield in mb.hist.items():
        for i, sample in enumerate(samples):
            hval = hfield.force(sample).val
            if key in plot_pd:
                hmax = plot_pd[key].get('max', None)
                hmin = plot_pd[key].get('min', None)
                bins = plot_pd[key].get('bins', 100)
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
    for sky in mb.plotting:

        plot = ift.Plot()
        sc = ift.StatCalculator()
        for sample in samples:
            sc.add(sample)
            m = mb.plotting[sky].force(sample)
            cmap = None
            if sky in plot_pd:
                if 'color' in plot_pd[sky]:
                    cmap = _get_cmap(plot_pd[sky]['color'])
            plot.add(m, cmap=cmap)
        plot.add(mb.plotting[sky].force(sc.mean), title='mean')
        plot.output(name=spath + sky + "_prior.png")


def data_plotting(data_dict, plot_pd, path):
    path += 'data/'
    if not os.path.exists(path):
        os.makedirs(path)
    for data in data_dict:
        if data[:9] == 'projected':
            plot = ift.Plot()
            if data in plot_pd:
                for zmax, zmin in zip(plot_pd[data]['max'], plot_pd[data]['min']):
                    plot.add(data_dict[data], title=data, vmax=zmax, vmin=zmin)
            else:
                plot.add(data_dict[data], title=data)
            ny = len(plot._plots)
            bd = plot_pd.get(data, {'t': None})
            if 'color' in bd:
                cmap = _get_cmap(bd['color'])
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


def power_plotting(amplitude_model_dict, sky_model_dict, samples, string, path):
    path += 'power/'
    if amplitude_model_dict == {}:
        return
    for amp in amplitude_model_dict:
        try:
            amp_model_samples = [amplitude_model_dict[amp].force(s) for s in samples]
            plot = ift.Plot()
            amp_path = path + amp + '/'
            if not os.path.exists(amp_path):
                os.makedirs(amp_path)
            linewidth = [1.] * len(amp_model_samples) + [3., ]
            plot.add(amp_model_samples,
                     title="Sampled Posterior Power Spectrum, " + amp, linewidth=linewidth)
            plot.output(name=amp_path + amp + string + ".png")
        except KeyError:
            continue
    for sky in sky_model_dict:
        if sky[:9] == 'projected':
            continue
        else:
            try:
                amp_path = path + sky + '/'
                if not os.path.exists(amp_path):
                    os.makedirs(amp_path)
                plot = ift.Plot()
                ht = ift.HarmonicTransformOperator(sky_model_dict[sky].target[0].get_default_codomain(),
                                                   sky_model_dict[sky].target[0])
                plot.add(
                    [ift.power_analyze(ht.adjoint(sky_model_dict[sky].force(s))) for s in samples],
                    title="Power Spectrum of Signal Posterior Samples, " + sky)
                plot.output(name=amp_path + sky + string + ".png")
            except KeyError:
                continue


def sky_map_plotting(sky_model_dict, samples, string, bounds, path):
    if sky_model_dict == {}:
        return
    path += 'sky/'
    for sky in sky_model_dict:
        try:
            plot = ift.Plot()
            sky_path = path + sky + '/'
            if not os.path.exists(sky_path):
                os.makedirs(sky_path)
            if isinstance(samples, list):
                sc = ift.StatCalculator()
                for sample in samples:
                    sc.add(sky_model_dict[sky].force(sample))
                m = sc.mean
            else:
                m = sky_model_dict[sky].force(samples)
            cmap, cmap_stddev = None, None
            if sky in bounds:
                if 'color' in bounds[sky]:
                    cmap = _get_cmap(bounds[sky]['color'])
                if 'color_stddev' in bounds[sky]:
                    cmap_stddev = _get_cmap(bounds[sky]['color_stddev'])
                for zmax, zmin, zmax_stddev, zmin_stddev in \
                        zip(bounds[sky]['max'], bounds[sky]['min'],
                            bounds[sky]['max_stddev'], bounds[sky]['min_stddev']):
                    plot.add(m, title="mean", vmax=zmax, vmin=zmin, cmap=cmap)
                    if len(samples) > 1:
                        plot.add(ift.sqrt(sc.var), title="std", vmax=zmax_stddev, vmin=zmin_stddev, cmap=cmap_stddev)
            else:
                plot.add(m, title="mean", colormap=cmap)
                if len(samples) > 1:
                    plot.add(ift.sqrt(sc.var), title="std", cmap=cmap_stddev)
        except KeyError:
            print('Key error in plot, concrning key {}'.format({sky}))
            continue
        if len(samples) == 1:
            nx = 1
            ny = len(plot._plots)
        else:
            nx = 2
            ny = int(len(plot._plots) / 2)
        plot.output(nx=nx, ny=ny, xsize=2 * 12, ysize=ny * 12, name=sky_path + sky + string + ".png")


def plot_error(noise_dict, data_dict, string, samples, bounds, path):
    path += 'noise/'
    for k in noise_dict.keys():
        try:
            noise_path = path + k + '/'
            if not os.path.exists(noise_path):
                os.makedirs(noise_path)
            sc = ift.StatCalculator()
            for sample in samples:
                sc.add(noise_dict[k].force((sample)))
            item_1 = np.log10(data_dict[k[len('stddev_estimate_'):] + '_stddev'])
            item_2 = np.log10(sc.mean.val)
            if k in bounds:
                xmin, xmax, ymin, ymax = bounds[k]
            else:
                xmin, xmax, ymin, ymax = 10 ** -5, 10 ** 5, 10 ** -5, 10 ** 5
            pl.figure()
            xxx, yyy, zzz = density_estimation(item_1, item_2, np.log10(xmin), np.log10(xmax),
                                               np.log10(ymin), np.log10(ymax), 100)
            xx = np.linspace(np.log10(xmin), np.log10(xmax), 10)
            yy = np.linspace(np.log10(ymin), np.log10(ymax), 10)

            pl.contour(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, linewidths=0.9, levels=np.linspace(0.01, 1, 10))
            c1 = pl.contourf(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, levels=np.linspace(0.01, 1, 10))
            col = pl.colorbar(c1)
            pl.scatter(item_1, item_2, marker=',', s=0.5, color='black')
            pl.plot(xx, yy, '--', c='red', linewidth=0.5)
            col.set_label(r'$\log\left(1+\mathcal{P}\right)$')
            pl.xlabel(r'measured $\sigma$')
            pl.ylabel(r'inferred $\sigma$')
            pl.xlim([np.log10(xmin), np.log10(xmax), ])
            pl.ylim([np.log10(ymin), np.log10(ymax), ])
            pl.savefig(noise_path + k + '_scatter_' + string + '.png', format='png', dpi=800)
            pl.close()
        except KeyError:
            continue


def _get_cmap(string):
    if string == 'niels':
        cmap = ncm.ncmap.fm()
    elif string == 'niels_stddev':
        cmap = ncm.ncmap.fu()
    else:
        cmap = string
    return cmap


def density_estimation(m1, m2, xmin, xmax, ymin, ymax, nbins):
    x, y = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([m1, m2])
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    return x, y, z
