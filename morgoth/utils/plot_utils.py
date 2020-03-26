import numpy as np
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import astropy.io.fits as fits
from gbmgeometry.utils.gbm_time import GBMTime
import astropy.time as astro_time
import requests
import shutil
from gbmgeometry.gbm_frame import GBMFrame
from gbmgeometry import gbm_detector_list
from astropy.coordinates import SkyCoord
import astropy.units as unit
import plotly
import plotly.graph_objs as go
import os
from gbm_drm_gen.io.balrog_healpix_map import BALROGHealpixMap
from threeML import *

from morgoth.utils.env import get_env_value
from morgoth.balrog_handlers import ProcessFitResults

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")

model_param_lookup = {
    'pl': ['ra (deg)', 'dec (deg)', 'K', 'index'],
    'cpl': ['ra (deg)', 'dec (deg)', 'K', 'index', 'xc'],
    'sbpl': ['ra (deg)', 'dec (deg)', 'K', 'alpha', 'break', 'beta'],
    'band': ['ra (deg)', 'dec (deg)', 'K', 'alpha', 'xp', 'beta'],
    'solar_flare': ['ra (deg)', 'dec (deg)', 'K-bl', 'xb-bl', 'alpha-bl', 'beta-bl', 'K-brems', 'Epiv-brems', 'kT-brems']
}


def create_corner_loc_plot(grb_name, report_type, version, datapath, model):
    """
    load fit results and create corner plots for ra and dec
    :return:
    """
    chain = loadtxt2d(datapath)

    # Get parameter for model
    parameter = model_param_lookup[model]

    # Check if loc at wrap at 360 degree
    # RA-DEC plot
    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(plot_hists=False, contour_labels='sigma',
                                                                           colors="#cd5c5c", flip=False)

    chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(None, None, None, None, color_p=True, blind=None)
    hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(chains[0], 'ra (deg)',
                                                                      'dec (deg)')  # ra, dec in deg here
    hist[hist == 0] = 1E-16
    val_contour = c1.plotter._convert_to_stdev(hist.T)

    # get list with all ra values that have a value of less than 0.99 asigned
    prob_list = []
    for val_array in val_contour.T:
        found = False
        for val in val_array:
            if val < 0.99:
                found = True
        prob_list.append(found)
    x_contour_prob = x_contour[prob_list]
    # if both side at wrap point at 2Pi have a value below 0.99 asigned we need to move the thing to get a decent plot
    if x_contour_prob[0] < 10 and x_contour_prob[-1] > 350:
        move = True
    else:
        move = False
    if move:
        for i in range(len(chain[:, 0])):
            if chain[i, 0] > 180:
                chain[i, 0] = chain[i, 0] - 360
    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(plot_hists=False, contour_labels='sigma',
                                                                           colors="#cd5c5c", flip=False, kde=2.0,
                                                                           max_ticks=5)

    filename = f'{base_dir}/{grb_name}/{report_type}/{version}/plots/{grb_name}_location_plot_{report_type}_{version}.png'

    c1.plotter.plot(filename=filename,
                    figsize="column")
