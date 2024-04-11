import nifty8 as ift
import libs as EgF

import numpy as np

"""
"""


def plot_prior(likelihood, n_samples, position=None, sky_models=None, power_models=None, scatter_pairs=None, plotting_kwargs=None):

    if sky_models is None:
        sky_models = dict()
    if power_models is None:
        power_models = dict()
    if scatter_pairs is None:
        scatter_pairs = dict()
    
    if position is None:
        samples = [ift.from_random(likelihood.domain) for _ in range(n_samples)]
    else:
        samples = [position + ift.from_random(likelihood.domain) for _ in range(n_samples)]

    plot_path='./runs/demo/'    
    for sky_name, sky in sky_models.items():
        EgF.sky_map_plotting(sky, samples, sky_name, plot_path, string='prior', plot_samples=True,
                                 **plotting_kwargs.get(sky_name, {}))
        if sky_name not in power_models:
            EgF.power_plotting(sky, samples, sky_name, plot_path, string='prior',
                                   from_power_model=False, **plotting_kwargs.get(sky_name, {}))
        if power_models is not None:
            for power_name, power in power_models.items():
                EgF.power_plotting(power, samples, power_name, plot_path, string='prior',
                               from_power_model=True,  **plotting_kwargs.get(power_name, {}))

        if scatter_pairs is not None:
            for key, (sc1, sc2) in scatter_pairs.items():
                EgF.scatter_plotting(sc1, sc2, key, plot_path, samples, string='prior',
                                 **plotting_kwargs.get(key, {}))


