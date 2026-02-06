import numpy as np
from astropy.io import fits
import nifty8 as ift
from nifty_cmaps import ncmap
import matplotlib.pyplot as plt


def plot_rmgal(params, dm, B, rm_gal, figname):

    plot = ift.Plot()
    plot.add(dm, vmin=0, vmax=500, title='DM [pc cm$^{-3}$]', cmap='magma', cmap_stddev=getattr(ncmap, 'fu')())
    plot.add(B, vmin=-2.50, vmax=2.50,  title='B [$\\mu$G], $\\gamma$=-3 ', cmap=getattr(ncmap, 'fu')(), cmap_stddev=getattr(ncmap, 'fu')())
    plot.add(rm_gal, vmin=-250, vmax=250, title='$\\phi_{gal}$ [rad m$^{-2}$]', cmap=getattr(ncmap, 'fm')(), cmap_stddev=getattr(ncmap, 'fu')())
    plot.output(name=f'{params["file_params.plot_path"]}{figname}', nx=1, ny=3) 



def plot_rmgalonly(params, gal, figname):

    plot = ift.Plot()
    plot.add(gal, vmin=-250, vmax=250)
    plot.output(name=f'{params["file_params.plot_path"]}{figname}')


def plot_rmeg(params, e_z, e_F, egal_contr, e_z_orig, e_F_orig_at_z, figname):

    fig, axs = plt.subplots(3, 2, figsize=(10,10))


    axs[2,1].set_xlabel('z')
    axs[2,0].set_xlabel('Stokes I (Jy)')

    axs[0,0].set_ylabel('Mock $\\phi_{eg}$ (rad/m$^2$)')
    axs[0,1].set_ylabel('Mock $\\phi_{eg}$ (rad/m$^2$)')
    axs[0,1].scatter(e_z, egal_contr, s=5, c='green')

    axs[2,1].hist(e_z_orig, bins=100, density=False, color='lightgrey')
    axs[1,1].hist(e_z, bins=100, density=False, color='green')

    axs[0,0].scatter(e_F, egal_contr, s=5, c='green')

    hist, bins = np.histogram(e_F_orig_at_z, bins=100)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    axs[2,0].hist(e_F_orig_at_z, bins=logbins,  density=False, color='lightgrey')

    hist, bins = np.histogram(e_F, bins=100)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    axs[1,0].hist(e_F, bins=logbins,  density=False, color='green')


    axs[2,0].set_xlim(0.0002,22000)
    axs[1,0].sharex(axs[2,0])
    axs[0,0].sharex(axs[2,0])
    axs[0,0].set_xscale('log')
    axs[1,0].set_xscale('log')
    axs[2,0].set_xscale('log')

    axs[0,0].set_ylim(-149,149)
    axs[1,0].set_ylim(0.1,1499)
    axs[1,1].set_ylim(0.1,2499)


    axs[0,1].set_xlim(-0.05,3.5)
    axs[1,1].sharex(axs[2,1])
    axs[0,1].sharex(axs[2,1])

    axs[0,1].set_ylim(-149,149)
    axs[2,0].set_ylim(0.1,119)
    axs[2,1].set_ylim(0.1,209)

    axs[2,0].set_ylabel('Observed #')
    axs[1,0].set_ylabel('Mock #')
    axs[2,1].set_ylabel('Observed #')
    axs[1,1].set_ylabel('Mock #')


    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.savefig(fname=f'{params["file_params.plot_path"]}{figname}', bbox_inches='tight')


def plot_mock(params, eg_data, noised_data, figname):
        
        plot = ift.Plot()
        plot.add(eg_data, vmin=-2.50, vmax=2.50)
        plot.add(noised_data, vmin=-2.50, vmax=2.50)
        plot.output(name=f'{params["file_params.plot_path"]}{figname}')

def plot_mock_vs_observed(params, dest_rm_eg, noised_rm_eg, dest_rm_gal, noised_rm_gal, figname):

        #questo plot in realtà ha senso farlo solo se si parte da un base_catalog osservato
        fig, axs = plt.subplots(1, 2)

        axs[1].set_xlabel('Observed Extragalactic RM (rad/m$^2$)')
        axs[1].set_ylabel('Simulated Extragalactic RM (rad/m$^2$)')
        axs[1].scatter(dest_rm_eg, noised_rm_eg)
        axs[0].set_xlabel('Observed Galactic RM ($rad/m^2$)')
        axs[0].set_ylabel('Simulated Galactic RM ($rad/m^2$)')
        axs[0].scatter(dest_rm_gal,noised_rm_gal)
        plt.savefig(fname=f'{params["file_params.plot_path"]}{figname}', bbox_inches='tight')