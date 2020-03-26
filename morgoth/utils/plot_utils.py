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


def create_corner_all_plot(grb_name, report_type, version, datapath, model):
    """
    load fit results and create corner plots for all parameters
    :return:
    """
    chain = loadtxt2d(datapath)

    # Get parameter for model
    parameter = model_param_lookup[model]

    # RA-DEC plot
    c2 = ChainConsumer()

    c2.add_chain(chain[:, :-1], parameters=parameter).configure(plot_hists=False, contour_labels='sigma',
                                                                colors="#cd5c5c", flip=False, max_ticks=3)

    filename = f'{base_dir}/{grb_name}/{report_type}/{version}/plots/{grb_name}_allcorner_plot_{report_type}_{version}.png'

    c2.plotter.plot(filename=filename,
                    figsize="column")


def mollweide_plot(grb_name, report_type, version, trigdat_file, post_equal_weigts_file, used_dets, model, ra, dec, swift=None):
    # get earth pointing in icrs and the pointing of dets in icrs

    with fits.open(trigdat_file) as f:
        quat = f['TRIGRATE'].data['SCATTITD'][0]
        sc_pos = f['TRIGRATE'].data['EIC'][0]
        times = f['TRIGRATE'].data['TIME'][0]

    # get a det object and calculate with this the position of the earth, the moon and the sun seen from the satellite
    # in the icrs system
    det_1 = gbm_detector_list[used_dets[-1]](quaternion=quat, sc_pos=sc_pos, time=astro_time.Time(utc(times)))
    earth_pos = det_1.earth_position_icrs
    sun_pos = det_1.sun_position_icrs
    moon_pos = det_1.moon_position_icrs
    # get pointing of all used dets
    det_pointing = {}
    for det_name in used_dets:
        det = gbm_detector_list[det_name](quaternion=quat, sc_pos=sc_pos)
        det_pointing[det_name] = det.det_ra_dec_icrs

    # set a figure with a hammer projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='hammer')

    # plot EARTH shadow
    ra_e = earth_pos.ra.rad
    dec_e = earth_pos.dec.rad
    if ra_e > np.pi:
        ra_e = ra_e - 2 * np.pi

    earth_opening = 67  # degree
    earth = FOV(ra_e, dec_e, earth_opening * np.pi / 180)
    if len(earth) == 2:
        ax.fill(earth[0], earth[1], 'b', alpha=0.2, label='EARTH')
    else:
        ax.fill(earth[0], earth[1], 'b', alpha=0.2, label='EARTH')
        ax.fill(earth[2], earth[3], 'b', alpha=0.2)

    # Plot GRB contours from fit
    # Get contours
    x_contour, y_contour, val_contour, x_contour_1, x_contour_2,\
    val_contour_1, val_contour_2 = get_contours(model, post_equal_weigts_file)

    if len(x_contour_1) > 0:
        ax.contourf(x_contour_1, y_contour, val_contour_1, levels=[0, 0.68268949, 0.9545],
                    colors=['navy', 'lightgreen'])
    if len(x_contour_2) > 0:
        ax.contourf(x_contour_2, y_contour, val_contour_2, levels=[0, 0.68268949, 0.9545],
                    colors=['navy', 'lightgreen'])

    # Plot GRB best fit
    ra_center = ra * np.pi / 180
    dec_center = dec * np.pi / 180
    if ra_center > np.pi:
        ra_center = ra_center - 2 * np.pi

    ax.scatter(ra_center, dec_center, label='Balrog Position', s=40, color='green', marker="*")
    ax.annotate(f'Balrog Position {grb_name}',
                xy=(ra_center, dec_center),  # theta, radius
                xytext=(0.55, 0.15),  # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.02, width=1, headwidth=5, headlength=5),
                horizontalalignment='left',
                verticalalignment='bottom',
                )

    # Plot 60 degree FOV of DETS
    color_dict = {'n0': 'blue', 'n1': 'navy', 'n2': 'crimson', 'n3': 'lightgreen', 'n4': 'orchid', 'n5': 'brown',
                  'n6': 'firebrick', 'n7': 'plum', 'n8': 'darkgreen', 'n9': 'olive', 'na': 'aqua',
                  'nb': 'darkorange', 'b0': 'darkmagenta', 'b1': 'indigo'}
    FOV_opening = 60  # degree
    for keys in det_pointing:
        pointing = det_pointing[keys]
        ra_d = pointing[0] * np.pi / 180
        dec_d = pointing[1] * np.pi / 180
        if ra_d > np.pi:
            ra_d = ra_d - 2 * np.pi
        name = str(keys)
        fov = FOV(ra_d, dec_d, FOV_opening * np.pi / 180)
        color = str(color_dict[keys])
        if len(fov) == 2:
            ax.plot(fov[0], fov[1], color=color, label=name, linewidth=0.5)
        else:
            ax.plot(fov[0], fov[1], color=color, label=name, linewidth=0.5)
            ax.plot(fov[2], fov[3], color=color, linewidth=0.5)

    # Plot Sun
    ra_s = sun_pos.ra.rad
    dec_s = sun_pos.dec.rad
    if ra_s > np.pi:
        ra_s = ra_s - 2 * np.pi
    ax.scatter(ra_s, dec_s, label='SUN', s=30, color='yellow')

    # MOON
    ra_m = moon_pos.ra.rad
    dec_m = moon_pos.dec.rad
    if ra_m > np.pi:
        ra_m = ra_m - 2 * np.pi
    ax.scatter(ra_m, dec_m, label='MOON', s=30, color='grey')

    # if we have a swift position plot it here
    if swift is not None:
        # Plot SWIFT position if there is one
        ra_swift = float(swift['ra']) * np.pi / 180
        dec_swift = float(swift['dec']) * np.pi / 180
        if ra_swift > np.pi:
            ra_swift = ra_swift - 2 * np.pi
        ax.scatter(ra_swift, dec_swift, label='SWIFT Position', s=40, marker="X", color='magenta', alpha=0.2)
        ax.annotate('SWIFT Position SWIFT-trigger {}'.format(swift['trigger']),
                    xy=(ra_swift, dec_swift),  # theta, radius
                    xytext=(0.55, 0.78),  # fraction, fraction
                    textcoords='figure fraction',
                    arrowprops=dict(facecolor='black', shrink=0.02, width=1, headwidth=5, headlength=5),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    )

    # set title, legend and grid
    plt.title(f'{grb_name} Position (J2000)', y=1.08)
    ax.grid()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

    # save figure
    save_path = f'{base_dir}/{grb_name}/{report_type}/{version}/plots/{grb_name}_molllocation_plot_{report_type}_{version}.png'

    fig.savefig(save_path, bbox_inches='tight', dpi=1000)


def azimuthal_plot_sat_frame(grb_name, report_type, version, trigdat_file, ra, dec):
    """
    plot azimuth plot in sat frame to check if burst comes from the solar panel sides
    :return:
    """
    ra_center = ra * np.pi / 180
    dec_center = dec * np.pi / 180
    if ra_center > np.pi:
        ra_center = ra_center - 2 * np.pi

    with fits.open(trigdat_file) as f:
        quat = f['TRIGRATE'].data['SCATTITD'][0]
        sc_pos = f['TRIGRATE'].data['EIC'][0]
        times = f['TRIGRATE'].data['TIME'][0]

    cone_opening = 45.0  # cone opening for solar panel side in deg
    loc_icrs = SkyCoord(ra=ra_center * 180 / np.pi, dec=dec_center * 180 / np.pi, unit='deg', frame="icrs")
    q1, q2, q3, q4 = quat
    scx, scy, scz = sc_pos
    loc_sat = loc_icrs.transform_to(GBMFrame(quaternion_1=q1,
                                             quaternion_2=q2,
                                             quaternion_3=q3,
                                             quaternion_4=q4,
                                             sc_pos_X=scx,
                                             sc_pos_Y=scy,
                                             sc_pos_Z=scz,
                                             ))
    ra_sat = Angle(loc_sat.lon.deg * unit.degree)
    dec_sat = Angle(loc_sat.lat.deg * unit.degree)
    ra_sat.wrap_at('180d', inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    # Fill area where the solar panels may cause a systematic error

    r_bound = np.arange(0, 200, 0.5)
    phi_bound = np.ones_like(r_bound)
    ax.fill_betweenx(r_bound, phi_bound * (np.pi / 2 - cone_opening * (np.pi / 180)),
                     phi_bound * (np.pi / 2 + cone_opening * (np.pi / 180)), color='grey', alpha=0.2,
                     label='solar panel sides')
    ax.fill_betweenx(r_bound, phi_bound * (-np.pi / 2 - cone_opening * (np.pi / 180)),
                     phi_bound * (-np.pi / 2 + cone_opening * (np.pi / 180)), color='grey', alpha=0.2)

    # Fill other area and label with b0 and b1 side
    ax.fill_betweenx(r_bound, phi_bound * (-np.pi / 2 + cone_opening * (np.pi / 180)),
                     phi_bound * (np.pi / 2 - cone_opening * (np.pi / 180)), color='lime', alpha=0.2,
                     label='b0 side')
    ax.fill_betweenx(r_bound, phi_bound * (-np.pi / 2 - cone_opening * (np.pi / 180)),
                     phi_bound * (np.pi / 2 + cone_opening * (np.pi / 180)), color='blue', alpha=0.2,
                     label='b1 side')

    # SAT coordinate system#
    ax.quiver(np.pi / 2, 0, 0, 1, scale=2.)
    ax.text((np.pi / 2) * 1.07, 0.9, "y")
    ax.quiver(0, 0, 1, 0, scale=2.)
    ax.text(-(np.pi / 2) * 0.07, 0.9, "x")
    ax.set_rlim((0, 1))
    ax.set_yticklabels([])

    # Plot Burst direction in Sat-Coord#
    phi_b = ra_sat.value * np.pi / 180
    u = np.cos(phi_b)
    v = np.sin(phi_b)

    q = ax.quiver(0, 0, u, v, scale=2., color='yellow', linewidth=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    ax.quiverkey(q, X=1.3, Y=0.5, U=0.4,
                 label=f'{grb_name}', labelpos='N')
    ax.set_title(f'{grb_name} direction in the sat. frame', y=1.08)

    save_path = f"{base_dir}/{grb_name}/{report_type}/{version}/plots/{grb_name}_satellite_plot_{report_type}_{version}.png"

    fig.savefig(save_path, bbox_inches='tight', dpi=1000)




def loadtxt2d(intext):
    try:
        return np.loadtxt(intext, ndmin=2)
    except:
        return np.loadtxt(intext)

def utc(met):
    """
    get utc time from met time
    :return:
    """
    time = GBMTime.from_MET(met)
    return time.time.fits

def seperation_smaller_angle(center, phi, theta, angle):
    """
    returns all phi theta combinations that have a smaler angle to center than the given angle
    :param phi:
    :param theta:
    :param angle:
    :return:
    """
    c1, c2, c3 = center
    sep = np.arccos(c1 * np.cos(theta) * np.cos(phi) + c2 * np.cos(theta) * np.sin(phi) + c3 * np.sin(theta))
    return phi[sep < angle], theta[sep < angle]

def FOV(center_ra, center_dec, angle):  # in rad!
    """
    calculate the points on a sphere inside the FOV of a det at center_ra, center_dec with a given viewing angle
    :param center_dec:
    :param angle:
    :return:
    """

    ra = center_ra
    dec = center_dec

    center = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    phi_bound_l = -np.pi
    phi_bound_h = np.pi

    phi_list = np.arange(phi_bound_l, phi_bound_h, (phi_bound_h - phi_bound_l) / 1000)
    theta_list = np.arange(-np.pi / 2, np.pi / 2, np.pi / 2000)
    phi, theta = np.meshgrid(phi_list, theta_list)
    phi, theta = seperation_smaller_angle(center, phi, theta, angle)

    phi_circle = []
    theta_max = []
    theta_min = []
    for phi_v in phi_list:
        phi_inter = phi[phi == phi_v]
        theta_inter = theta[phi == phi_v]
        if len(phi_inter) > 0:
            phi_circle.append(phi_v)
            theta_max.append(np.max(theta_inter))
            theta_min.append(np.min(theta_inter))

    phi_circle = np.array(phi_circle)
    theta_max = np.array(theta_max)
    theta_min = np.array(theta_min)
    # get index if values in phi_circle change by more than 10 degree
    i = 0
    split_index = None
    while i < len(phi_circle) - 1:
        if phi_circle[i + 1] > phi_circle[i] + 10 * np.pi / 180:
            split_index = i + 1
        i += 1

    if split_index is not None:
        phi_circle_0 = phi_circle[:split_index]
        phi_circle_1 = phi_circle[split_index:]
        theta_min_0 = theta_min[:split_index]
        theta_min_1 = theta_min[split_index:]
        theta_max_0 = theta_max[:split_index]
        theta_max_1 = theta_max[split_index:]

        phi_circle_0 = np.concatenate((phi_circle_0, np.flip(phi_circle_0, 0), np.array([phi_circle_0[0]])))
        phi_circle_1 = np.concatenate((phi_circle_1, np.flip(phi_circle_1, 0), np.array([phi_circle_1[0]])))
        theta_circle_0 = np.concatenate((theta_min_0, np.flip(theta_max_0, 0), np.array([theta_min_0[0]])))
        theta_circle_1 = np.concatenate((theta_min_1, np.flip(theta_max_1, 0), np.array([theta_min_1[0]])))
        return [phi_circle_0, theta_circle_0, phi_circle_1, theta_circle_1]
    else:
        phi_circle = np.concatenate((phi_circle, np.flip(phi_circle, 0), np.array([phi_circle[0]])))
        theta_circle = np.concatenate((theta_min, np.flip(theta_max, 0), np.array([theta_min[0]])))
        return [phi_circle, theta_circle]

def xyz(phi, theta):
    """
    gives xyz of point on unit sphere defined by theta and phi
    :param phi:
    :param theta:
    :return:
    """
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z

def phi_0(theta, ra_c, dec_c, rad_r):
    """
    Gives the corresponding phi value to the given theta for which the separation to ra_c, dec_c is the angle rad_r
    This has two solutions. So phi_0 and phi_1 corresponse to the two solutions.
    :param theta:
    :param ra_c:
    :param dec_c:
    :param rad_r:
    :return:
    """
    detx = np.cos(ra_c) * np.cos(dec_c)
    dety = np.sin(ra_c) * np.cos(dec_c)
    detz = np.sin(dec_c)
    a = detx * np.cos(theta)
    b = dety * np.cos(theta)
    c = detz * np.sin(theta)
    d = np.cos(rad_r)
    z = (-c + (a ** 2 * c / (a ** 2 + b ** 2)) + d - (a ** 2 * d / (a ** 2 + b ** 2)) + (
            a * np.sqrt(-b ** 2 * (-a ** 2 - b ** 2 + c ** 2 - 2 * c * d + d ** 2))) / (a ** 2 + b ** 2)) / b
    n = (-a * c + a * d - np.sqrt(
        a ** 2 * b ** 2 + b ** 4 - b ** 2 * c ** 2 + 2 * b ** 2 * c * d - b ** 2 * d ** 2)) / (a ** 2 + b ** 2)
    phi = np.arctan2(z, n)
    return phi

def phi_1(theta, ra_c, dec_c, rad_r):
    """
   Gives the corresponding phi value to the given theta for which the separation to ra_c, dec_c is the angle rad_r
   This has two solutions. So phi_0 and phi_1 corresponse to the two solutions.
   :param theta:
   :param ra_c:
   :param dec_c:
   :param rad_r:
   :return:
   """
    detx = np.cos(ra_c) * np.cos(dec_c)
    dety = np.sin(ra_c) * np.cos(dec_c)
    detz = np.sin(dec_c)
    a = detx * np.cos(theta)
    b = dety * np.cos(theta)
    c = detz * np.sin(theta)
    d = np.cos(rad_r)
    z = (-c + (a ** 2 * c / (a ** 2 + b ** 2)) + d - (a ** 2 * d / (a ** 2 + b ** 2)) - (
            a * np.sqrt(-b ** 2 * (-a ** 2 - b ** 2 + c ** 2 - 2 * c * d + d ** 2))) / (a ** 2 + b ** 2)) / b
    n = (-a * c + a * d + np.sqrt(
        a ** 2 * b ** 2 + b ** 4 - b ** 2 * c ** 2 + 2 * b ** 2 * c * d - b ** 2 * d ** 2)) / (a ** 2 + b ** 2)
    phi = np.arctan2(z, n)
    return phi


def get_contours(model, post_equal_weigts_file):
    # Get parameter for model
    parameter = model_param_lookup[model]

    # get contours
    chain = loadtxt2d(post_equal_weigts_file)

    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(plot_hists=False, contour_labels='sigma',
                                                                           colors="#cd5c5c", flip=False)

    # x_contour, y_contour, val_contour = c1.plotter.get_contours_list('ra (deg)', 'dec (deg)')

    chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(None, None, None, None, color_p=True, blind=None)
    hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(chains[0], 'ra (deg)',
                                                                      'dec (deg)')  # ra, dec in deg here
    hist[hist == 0] = 1E-16
    val_contour = c1.plotter._convert_to_stdev(hist.T)

    # Check if at wrap point because then we have to shift this to avoid an ugly space at ra=0 in the plot
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
                                                                               colors="#cd5c5c", flip=False)

        # x_contour, y_contour, val_contour = c1.plotter.get_contours_list('ra (deg)', 'dec (deg)')

        chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(None, None, None, None, color_p=True, blind=None)
        hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(chains[0], 'ra (deg)',
                                                                          'dec (deg)')  # ra, dec in deg here
        hist[hist == 0] = 1E-16
        val_contour = c1.plotter._convert_to_stdev(hist.T)

    x_contour = x_contour * np.pi / 180
    y_contour = y_contour * np.pi / 180

    # split in ra area between 0 and pi & pi and 2 pi and wrap the second one to -pi to 0
    x_contour_1 = x_contour[x_contour < np.pi]
    x_contour_2 = x_contour[x_contour > np.pi] - 2 * np.pi

    val_contour_1 = val_contour[:, x_contour < np.pi]
    val_contour_2 = val_contour[:, x_contour > np.pi]

    return x_contour, y_contour, val_contour, x_contour_1, x_contour_2, val_contour_1, val_contour_2