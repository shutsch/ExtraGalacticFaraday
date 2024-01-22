import os
import numpy as np
import nifty8 as ift
from .nifty_cmaps import ncmap
import matplotlib.pyplot as pl
from matplotlib import cm


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


def scatter_plotting(model_1, model_2, name, path, plot_obj=None, string=None, **kwargs):
    if string is None:
        string = ''
    path += 'scatter/'

    scatter_path = path + name + '/'
    if not os.path.exists(scatter_path):
        os.makedirs(scatter_path)

    if isinstance(model_1, ift.Operator) and not isinstance(model_1, ift.Field):
        if isinstance(plot_obj, list):
            sc_1 = ift.StatCalculator()
            for sample in plot_obj:
                sc_1.add(model_1.force(sample))
            val_1 = sc_1.mean
        elif isinstance(plot_obj, ift.Field) or isinstance(plot_obj, ift.MultiField):
            val_1 = model_1.force(plot_obj)
        else:
            raise TypeError
    else:
        val_1 = model_1

    if isinstance(model_2, ift.Operator) and not isinstance(model_2, ift.Field):
        if isinstance(plot_obj, list):
            sc_2 = ift.StatCalculator()
            for sample in plot_obj:
                sc_2.add(model_2.force(sample))
            val_2 = sc_2.mean
        elif isinstance(plot_obj, ift.Field) or isinstance(plot_obj, ift.MultiField):
            val_2 = model_2.force(plot_obj)
        else:
            raise TypeError
    else:
        val_2 = model_2
    val_1 = val_1.val if isinstance(val_1, ift.Field) else val_1
    val_2 = val_2.val if isinstance(val_2, ift.Field) else val_2
    #xmax, xmin = bounds.get(key_1, (val_1.max(), val_1.min(), ))
    xmax, xmin = (val_1.max(), val_1.min(),)
    ymax, ymin = (val_2.max(), val_2.min(), )

    pl.figure()
    xxx, yyy, zzz = _density_estimation(val_1, val_2, xmin, xmax, ymin, ymax, 100)
    xx = np.linspace(xmin, xmax, 10)
    yy = np.linspace(ymin, ymax, 10)

    pl.contour(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, linewidths=0.9,
               #levels=np.linspace(0.01, 1, 10)
               )
    c1 = pl.contourf(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool,
        # levels=np.linspace(0.01, 1, 10)
                     )
    col = pl.colorbar(c1)
    col.set_label(kwargs.get('c_label', None))
    pl.scatter(val_1, val_2, marker=',', s=0.5, color='black')
    pl.plot(xx, yy, '--', c='red', linewidth=0.5)
    pl.xlabel(kwargs.get('x_label', None))
    pl.ylabel(kwargs.get('y_label', None))
    pl.xlim([xmin, xmax, ])
    pl.ylim([ymin, ymax, ])
    pl.savefig(scatter_path + name + '_' + string + '.png', format='png', dpi=800)
    pl.close()


def power_plotting(model, samples, name, path, from_power_model, string=None, **kwargs):
    if string is None:
        string = ''
    amp_path = path + 'power/' + name + '/'
    if not os.path.exists(amp_path):
        os.makedirs(amp_path)
    if from_power_model:
        amp_model_samples = [model.force(s) for s in samples]
        linewidth = [1.] * len(amp_model_samples) + [3., ]
        alpha = [.5] * len(amp_model_samples) + [1., ]
        color = kwargs.get('color', 'green')
        color = len(amp_model_samples) * [color,] + ['black']
        amp_model_samples.append(sum(amp_model_samples)/len(amp_model_samples))

        plo = ift.Plot()
        plo.add(amp_model_samples, title="Sampled Posterior Power Spectrum, " + name, linewidth=linewidth, alpha=alpha,
                color=color)
        plo.output(name=amp_path + name + '_' + string + ".png")

    else:
        plo = ift.Plot()
        ht = ift.HarmonicTransformOperator(model.target[0].get_default_codomain(), model.target[0])
        amp_model_samples = [ift.power_analyze(ht.adjoint(model.force(s))) for s in samples]
        linewidth = [1.] * len(amp_model_samples) + [3., ]
        alpha = [.5] * len(amp_model_samples) + [1., ]
        color = kwargs.get('color', 'green')
        color = len(amp_model_samples) * [color,] + ['black']
        amp_model_samples.append(sum(amp_model_samples)/len(amp_model_samples))

        plo.add(amp_model_samples, title="Calculated Power Spectrum, " + name, linewidth=linewidth, alpha=alpha,
                color=color)
        plo.output(name=amp_path + name + '_' + string + ".png")


def sky_map_plotting(model, plot_obj, name, path, string=None, **kwargs):
    if string is None:
        string = ''
    sky_path = path + 'sky/' + name + '/'
    if not os.path.exists(sky_path):
        os.makedirs(sky_path)
    plot = ift.Plot()
    if isinstance(plot_obj, list):
        sc = ift.StatCalculator()
        for sample in plot_obj:
            sc.add(model.force(sample))
        m = sc.mean
    else:
        m = model.force(plot_obj)
    if 'cmap' in kwargs:
        try:
            kwargs['cmap'] = getattr(ncmap, kwargs['cmap'])()
            kwargs['vmin']= kwargs['vmin_mean']
            kwargs['vmax']= kwargs['vmax_mean']
        except AttributeError:
            kwargs['cmap'] = getattr(cm, kwargs['cmap'])

    plot.add(m, title="mean", **kwargs)
    if len(plot_obj) > 1:
        if 'cmap_stddev' in kwargs:
            try:
                kwargs['cmap'] = getattr(ncmap, kwargs['cmap_stddev'])()
                kwargs['vmin']= kwargs['vmin_std']
                kwargs['vmax']= kwargs['vmax_std']
            except AttributeError:
                kwargs['cmap'] = getattr(cm, kwargs['cmap_stddev'])
        plot.add(ift.sqrt(sc.var), **kwargs, title='std')
    if len(plot_obj) == 1:
        nx = 1
        ny = len(plot._plots)
    else:
        nx = 2
        ny = int(len(plot._plots) / 2)
    plot.output(nx=nx, ny=ny, xsize=2 * 12, ysize=ny * 12, name=sky_path + name + '_' + string + ".png")


def _density_estimation(m1, m2, xmin, xmax, ymin, ymax, nbins):
    x, y = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([m1, m2])
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    return x, y, z
