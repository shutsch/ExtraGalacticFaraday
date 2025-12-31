import os
import numpy as np
import nifty8 as ift
from .nifty_cmaps import ncmap
from astropy.modeling.models import Gaussian1D
from healpy.newvisufunc import projview
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg') 



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


def scatter_plotting_posterior(model_1, model_2, name, path, plot_obj=None, string=None, **kwargs):
    if string is None:
        string = ''
    path += 'scatter/'

    scatter_path = path + name + '/'
    if not os.path.exists(scatter_path):
        os.makedirs(scatter_path)

    if isinstance(model_1, ift.Operator) and not isinstance(model_1, ift.Field):
        if isinstance(plot_obj, list):
            val_1_list = list()
            print(*plot_obj)
            for s in plot_obj:
                val_1_list.append(model_1.force(s).val)
            val_1=np.array(val_1_list)
        else:
            raise TypeError
    else:
        val_1 = model_1

    if isinstance(model_2, ift.Operator) and not isinstance(model_2, ift.Field):
        if isinstance(plot_obj, list):
            val_2_list = list()
            for s in plot_obj:
                val_2_list.append(model_2.force(s).val)
            val_2=np.array(val_2_list)
        else:
            raise TypeError
    else:
        val_2 = model_2
    val_1 = val_1.val if isinstance(val_1, ift.Field) else val_1
    val_2 = val_2.val if isinstance(val_2, ift.Field) else val_2
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
    p=open('output_plt.txt', 'w')
    pl.savefig(scatter_path + name + '_' + string + '.png', format='png', dpi=800)
    pl.close()
    
    p.write('val_1'+str(val_1)+'val_2'+str(val_2))
    p.close()

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

    if isinstance(plot_obj, list):
        sc = ift.StatCalculator()
        for sample in plot_obj:
            sc.add(model.force(sample))
        m = sc.mean
        print(m.val)
    else:
        m = model.force(plot_obj)
    if 'cmap' in kwargs:
        try:
            kwargs['cmap'] = getattr(ncmap, kwargs['cmap'])()
            kwargs['vmin']= kwargs['vmin_mean']
            kwargs['vmax']= kwargs['vmax_mean']
        except AttributeError:
            kwargs['cmap'] = getattr(cm, kwargs['cmap'])
    projview(m.val, sub=121, coord=["G"], flip="astro", projection_type="mollweide", cmap=kwargs['cmap'], min=kwargs['vmin'], max=kwargs['cmap'], title='Mean (rad m$^{-2}$)', fontsize={'title':10, 'cbar_tick_label':10})
 
    if 'cmap_stddev' in kwargs:
        if 'cmap_stddev' in kwargs:
            try:
                kwargs['cmap'] = getattr(ncmap, kwargs['cmap_stddev'])()
                kwargs['vmin']= kwargs['vmin_std']
                kwargs['vmax']= kwargs['vmax_std']
            except AttributeError:
                kwargs['cmap'] = getattr(cm, kwargs['cmap_stddev'])
    projview(ift.sqrt(sc.var).val, sub=122, coord=["G"], flip="astro", projection_type="mollweide", cmap=kwargs['cmap'], min=kwargs['vmin'], max=kwargs['cmap'], title='Uncertainty (rad m$^{-2}$)', fontsize={'title':10, 'cbar_tick_label':10})

    pl.savefig(sky_path + name + '_' + string + ".png", bbox_inches='tight')
    pl.close()


def sky_map_plotting_seb(model, plot_obj, name, path, string=None, **kwargs):
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
        print(m.val)
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

def density_plot(params,axs,x,y,mx,my,sx,sy, width, points, axsx, axsy, xlabel=None,ylabel=None):
    xxx, yyy, zzz = _density_estimation(x, y, mx-width*sx,mx+width*sx, my-width*sy,my+width*sy, points)
    axs[axsx,axsy].imshow(np.rot90(zzz), cmap=plt.cm.gist_earth_r, extent=[mx-width*sx,mx+width*sx, my-width*sy,my+width*sy], aspect="auto")
    axs[axsx,axsy].set_ylim(my-width*sy,my+width*sy)
    axs[axsx,axsy].set_xlim(mx-width*sx,mx+width*sx)
    axs[axsx,axsy].set_xlabel(f'{xlabel}', fontsize = params['plot.fontsize']) if xlabel is not None else '' 
    axs[axsx,axsy].set_ylabel(f'{ylabel}', fontsize = params['plot.fontsize']) if ylabel is not None else '' 


    z=np.ravel(np.array(zzz))
    order=np.argsort(z)
    zsorted=z[order]
    Z=np.cumsum(zsorted)
    Z /=Z[-1]
    levels=np.searchsorted(Z,1-np.array([0.997,0.95,0.68]),side='left')
    new_levels=zsorted[levels]
    axs[axsx,axsy].contour(xxx,yyy,zzz,levels=new_levels,colors=np.array(['brown','brown', 'brown']), linestyles='-', alpha=0.5)
    axs[axsx,axsy].scatter(x, y, color='k', s=params['plot.markersize'])

def histo_plot(params, axs, x, mx, sx, width, axsx, axsy, xlabel=None):
    axs[axsx,axsy].hist(x, bins=params['plot.bins'], color='lightgray')
    axs[axsy,axsy].tick_params('y', labelleft=False)
    axs[axsx,axsy].set_xlim(mx-width*sx,mx+width*sx)
    axs[axsx,axsy].set_xlabel(f'{xlabel}', fontsize = params['plot.fontsize']) if xlabel is not None else '' 


def gauss_plot(params, axs, mx, sx, width, axsx, axsy, points, label=None):
    x = np.linspace(mx-width*sx,mx+width*sx, points)
    y = Gaussian1D(amplitude=params['plot.amplitude'], mean=params['prior_mean.prior_mean_int'], stddev= params['prior_std.prior_std_int'])
    axs[axsx,axsy].plot(x, y(x), 'b-', label=f'{label}') if label is not None else axs[axsx,axsy].plot(x, y(x), 'b-', alpha=0.5) 

def sigma_plot(params, axs, mx, sx, width, color, label, axsx, axsy):
    axs[axsx, axsy].axvline(x = mx+width[0]*sx, color = color[0], linestyle='--', alpha=0.5, label=f'{label[0]}')
    axs[axsx, axsy].axvline(x = mx-width[0]*sx, color = color[0], linestyle='--', alpha=0.5)
    axs[axsx, axsy].axvline(x = mx+width[1]*sx, color = color[1], linestyle='--', alpha=0.5, label=f'{label[1]}')
    axs[axsx, axsy].axvline(x = mx-width[1]*sx, color = color[1], linestyle='--', alpha=0.5)
    axs[axsx, axsy].axvline(x = mx+width[2]*sx, color = color[2], linestyle='--', alpha=0.5, label=f'{label[2]}')
    axs[axsx, axsy].axvline(x = mx-width[2]*sx, color = color[2], linestyle='--', alpha=0.5)

def noise_plot(params, sigma1, sigma2, sigma1_mock, sigma2_mock, figname):
    fig, axs = plt.subplots(2, 2)

    axs[0,0].hist(sigma1, bins=100, density=True, color='green')
    
    axs[0,1].hist(sigma2, bins=100, density=True, color='green')

    axs[1,0].hist(sigma1_mock, bins=100, density=True, color='lightgrey')
    axs[1,0].set_xlabel('$\\sigma_{1}$ (rad/m$^2$)')

    axs[1,1].hist(sigma2_mock, bins=100, density=True, color='lightgrey')
    axs[1,1].set_xlabel('$\\sigma_{2}$ (rad/m$^2$)')

    axs[1,1].sharex(axs[0,1])
    axs[1,0].sharex(axs[0,0])

    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])



    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.savefig(f'{params["file_params.plot_path"]}{figname}', bbox_inches='tight')

def eta_plotting(name, plot_obj, path, sigma_rm, gal_pos, mock_npi_indices, deviation, string=None, **kwargs):
    if string is None:
        string = ''
    eta_path = path + 'eta/'+ '/'
    if not os.path.exists(eta_path):
        os.makedirs(eta_path)
    plot = ift.Plot()
    if isinstance(plot_obj, list):
        sc = ift.StatCalculator()
        for sample in plot_obj:
            sc.add(sample[name])
        m = sc.mean
        print(m.val)
    else:
        m =plot_obj
    plot.add(m, title="mean", **kwargs)
    plot.output(name=eta_path + name +'_' + string + ".png")
    print('eta mean min', m.val.min())
    print('eta mean max', m.val.max())


    gal_pos=gal_pos.mask.astype('float64')
    gal_pos[gal_pos == 1.0] = m.val

    sigma_rm2=sigma_rm**2
    sigma_rm_corr2=gal_pos*sigma_rm2


    pl.clf()
    fig, ax = pl.subplots()
    print(gal_pos.size)
    print(deviation.size)
    ratio=deviation[np.where(gal_pos > 5.0)[0]]/np.sqrt(sigma_rm_corr2[np.where(gal_pos > 5.0)[0]])
    np.save(eta_path + name +'_deviation_' + string + ".npy", ratio)
    ax.hist(ratio, bins=100,  density=False, color='green')
    pl.tight_layout()
    pl.savefig(eta_path + name +'_deviation_' + string + ".png", dpi=300)


    with open(eta_path + name +'_summary_' + string + ".txt", 'w') as f:
        print('Number of eta >10', np.where(gal_pos>10.0)[0].size, file=f)
        print('Number of eta >5', np.where(gal_pos>5.0)[0].size, file=f)



   
    pl.clf()
    fig, ax = pl.subplots()
    ax.set_xlabel('$\\sigma^2_{RM}$')
    ax.set_ylabel('$\\sigma^2_{RM, corr}$')


    #xxx, yyy, zzz = _density_estimation(sigma_rm2, sigma_rm_corr2, sigma_rm2.min(), sigma_rm2.max(), sigma_rm_corr2.min(), sigma_rm_corr2.max(), 100)
    #ax.imshow(np.rot90(zzz), cmap=pl.cm.inferno, extent=[sigma_rm2.min(), sigma_rm2.max(), sigma_rm_corr2.min(), sigma_rm_corr2.max()], aspect="auto")
    ax.scatter(sigma_rm2, sigma_rm_corr2)
    ax.plot(sigma_rm2, sigma_rm2, color='lightsteelblue')
    #ax.set_xlim(sigma_rm2.min(), sigma_rm2.max())
    #ax.set_ylim(sigma_rm_corr2.min(), sigma_rm_corr2.max())
    ax.set_xscale("log")
    ax.set_yscale("log")
    pl.savefig(eta_path + name +'_sigma_' + string + ".png")


    pl.clf()
    fig=pl.figure(figsize=(40,1))
    print('gal_pos min', gal_pos.min())
    print('gal_pos max', gal_pos.max())

    pl.vlines(mock_npi_indices, ymin=0, ymax=gal_pos.max(), lw=0.005, color='k')
    pl.bar(np.arange(0,gal_pos.size,1), gal_pos)
    print('mock size', mock_npi_indices.size)
    pl.tight_layout()
    pl.savefig(eta_path + name +'_indices_' + string + ".png", dpi=300)

