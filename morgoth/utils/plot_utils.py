import os

import astropy.io.fits as fits
import astropy.time as astro_time
import astropy.units as unit
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from chainconsumer import ChainConsumer
from gbm_drm_gen.io.balrog_healpix_map import BALROGHealpixMap
from gbmgeometry import gbm_detector_list
from gbmgeometry.gbm_frame import GBMFrame
from gbmgeometry.utils.gbm_time import GBMTime

import morgoth.utils.file_utils as file_utils
from morgoth.utils.env import get_env_value

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")

_gbm_detectors = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]

model_param_lookup = {
    "pl": ["ra (deg)", "dec (deg)", "K", "index"],
    "cpl": ["ra (deg)", "dec (deg)", "K", "index", "xc"],
    "sbpl": ["ra (deg)", "dec (deg)", "K", "alpha", "break", "beta"],
    "band": ["ra (deg)", "dec (deg)", "K", "alpha", "xp", "beta"],
    "solar_flare": [
        "ra (deg)",
        "dec (deg)",
        "K-bl",
        "xb-bl",
        "alpha-bl",
        "beta-bl",
        "K-brems",
        "Epiv-brems",
        "kT-brems",
    ],
}


def create_corner_loc_plot(post_equal_weights_file, model, save_path):
    """
    load fit results and create corner plots for ra and dec
    :return:
    """
    chain = loadtxt2d(post_equal_weights_file)

    # Get parameter for model
    parameter = model_param_lookup[model]

    # Check if loc at wrap at 360 degree
    # RA-DEC plot
    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(
        plot_hists=False, contour_labels="sigma", colors="#cd5c5c", flip=False
    )

    chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(
        None, None, None, None, color_p=True, blind=None
    )
    hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(
        chains[0], "ra (deg)", "dec (deg)"
    )  # ra, dec in deg here
    hist[hist == 0] = 1e-16
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
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(
        plot_hists=False,
        contour_labels="sigma",
        colors="#cd5c5c",
        flip=False,
        kde=2.0,
        max_ticks=5,
    )

    file_utils.if_dir_containing_file_not_existing_then_make(save_path)

    c1.plotter.plot(filename=save_path, figsize="column")


def create_corner_all_plot(post_equal_weights_file, model, save_path):
    """
    load fit results and create corner plots for all parameters
    :return:
    """
    chain = loadtxt2d(post_equal_weights_file)

    # Get parameter for model
    parameter = model_param_lookup[model]

    # RA-DEC plot
    c2 = ChainConsumer()

    c2.add_chain(chain[:, :-1], parameters=parameter).configure(
        plot_hists=False,
        contour_labels="sigma",
        colors="#cd5c5c",
        flip=False,
        max_ticks=3,
    )

    file_utils.if_dir_containing_file_not_existing_then_make(save_path)
    c2.plotter.plot(filename=save_path, figsize="column")


def mollweide_plot(
    grb_name,
    trigdat_file,
    post_equal_weights_file,
    used_dets,
    model,
    ra,
    dec,
    save_path,
    swift=None,
):
    # get earth pointing in icrs and the pointing of dets in icrs

    with fits.open(trigdat_file) as f:
        quat = f["TRIGRATE"].data["SCATTITD"][0]
        sc_pos = f["TRIGRATE"].data["EIC"][0]
        times = f["TRIGRATE"].data["TIME"][0]

    # get a det object and calculate with this the position of the earth, the moon and the sun seen from the satellite
    # in the icrs system
    det_1 = gbm_detector_list[_gbm_detectors[used_dets[-1]]](
        quaternion=quat, sc_pos=sc_pos, time=astro_time.Time(utc(times))
    )
    earth_pos = det_1.earth_position_icrs
    sun_pos = det_1.sun_position_icrs
    moon_pos = det_1.moon_position_icrs
    # get pointing of all used dets
    det_pointing = {}
    for det_number in used_dets:
        det = gbm_detector_list[_gbm_detectors[det_number]](
            quaternion=quat, sc_pos=sc_pos
        )
        det_pointing[_gbm_detectors[det_number]] = det.det_ra_dec_icrs

    # set a figure with a hammer projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="hammer")

    # plot EARTH shadow
    ra_e = earth_pos.ra.rad
    dec_e = earth_pos.dec.rad
    if ra_e > np.pi:
        ra_e = ra_e - 2 * np.pi

    earth_opening = 67  # degree
    earth = FOV(ra_e, dec_e, earth_opening * np.pi / 180)
    if len(earth) == 2:
        ax.fill(earth[0], earth[1], "b", alpha=0.2, label="EARTH")
    else:
        ax.fill(earth[0], earth[1], "b", alpha=0.2, label="EARTH")
        ax.fill(earth[2], earth[3], "b", alpha=0.2)

    # Plot GRB contours from fit
    # Get contours
    (
        x_contour,
        y_contour,
        val_contour,
        x_contour_1,
        x_contour_2,
        val_contour_1,
        val_contour_2,
    ) = get_contours(model, post_equal_weights_file)

    if len(x_contour_1) > 0:
        ax.contourf(
            x_contour_1,
            y_contour,
            val_contour_1,
            levels=[0, 0.68268949, 0.9545],
            colors=["navy", "lightgreen"],
        )
    if len(x_contour_2) > 0:
        ax.contourf(
            x_contour_2,
            y_contour,
            val_contour_2,
            levels=[0, 0.68268949, 0.9545],
            colors=["navy", "lightgreen"],
        )

    # Plot GRB best fit
    ra_center = ra * np.pi / 180
    dec_center = dec * np.pi / 180
    if ra_center > np.pi:
        ra_center = ra_center - 2 * np.pi

    ax.scatter(
        ra_center, dec_center, label="Balrog Position", s=40, color="green", marker="*"
    )
    ax.annotate(
        f"Balrog Position {grb_name}",
        xy=(ra_center, dec_center),  # theta, radius
        xytext=(0.55, 0.15),  # fraction, fraction
        textcoords="figure fraction",
        arrowprops=dict(
            facecolor="black", shrink=0.02, width=1, headwidth=5, headlength=5
        ),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # Plot 60 degree FOV of DETS
    color_dict = {
        "n0": "blue",
        "n1": "navy",
        "n2": "crimson",
        "n3": "lightgreen",
        "n4": "orchid",
        "n5": "brown",
        "n6": "firebrick",
        "n7": "plum",
        "n8": "darkgreen",
        "n9": "olive",
        "na": "aqua",
        "nb": "darkorange",
        "b0": "darkmagenta",
        "b1": "indigo",
    }
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
    ax.scatter(ra_s, dec_s, label="SUN", s=30, color="yellow")

    # MOON
    ra_m = moon_pos.ra.rad
    dec_m = moon_pos.dec.rad
    if ra_m > np.pi:
        ra_m = ra_m - 2 * np.pi
    ax.scatter(ra_m, dec_m, label="MOON", s=30, color="grey")

    # if we have a swift position plot it here
    if swift is not None:
        # Plot SWIFT position if there is one
        ra_swift = float(swift["ra"]) * np.pi / 180
        dec_swift = float(swift["dec"]) * np.pi / 180
        if ra_swift > np.pi:
            ra_swift = ra_swift - 2 * np.pi
        ax.scatter(
            ra_swift,
            dec_swift,
            label="SWIFT Position",
            s=40,
            marker="X",
            color="magenta",
            alpha=0.2,
        )
        ax.annotate(
            "SWIFT Position SWIFT-trigger {}".format(swift["trigger"]),
            xy=(ra_swift, dec_swift),  # theta, radius
            xytext=(0.55, 0.78),  # fraction, fraction
            textcoords="figure fraction",
            arrowprops=dict(
                facecolor="black", shrink=0.02, width=1, headwidth=5, headlength=5
            ),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # set title, legend and grid
    plt.title(f"{grb_name} Position (J2000)", y=1.08)
    ax.grid()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

    # save figure
    file_utils.if_dir_containing_file_not_existing_then_make(save_path)
    fig.savefig(save_path, bbox_inches="tight", dpi=1000)


def brightobjects_plot( 
    grb_name,
    trigdat_file,
    post_equal_weights_file,
    used_dets,
    model,
    ra,
    dec,
    save_path,
    bright_sources,
    SGRs,
    swift=None,
):
    # get earth pointing in icrs and the pointing of dets in icrs
    with fits.open(trigdat_file) as f:
        quat = f["TRIGRATE"].data["SCATTITD"][0]
        sc_pos = f["TRIGRATE"].data["EIC"][0]
        times = f["TRIGRATE"].data["TIME"][0]

    # get a det object and calculate with this the position of the earth from the satellite
    # in the icrs system
    det_1 = gbm_detector_list[_gbm_detectors[used_dets[-1]]](
        quaternion=quat, sc_pos=sc_pos, time=astro_time.Time(utc(times))
    )
    earth_pos = det_1.earth_position_icrs
    # get pointing of all used dets
    det_pointing = {}
    for det_number in used_dets:
        det = gbm_detector_list[_gbm_detectors[det_number]](
            quaternion=quat, sc_pos=sc_pos
        )
        det_pointing[_gbm_detectors[det_number]] = det.det_ra_dec_icrs

    # set a figure with a hammer projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="hammer")

    # plot EARTH shadow
    ra_e = earth_pos.ra.rad
    dec_e = earth_pos.dec.rad
    if ra_e > np.pi:
        ra_e = ra_e - 2 * np.pi

    earth_opening = 67  # degree
    earth = FOV(ra_e, dec_e, earth_opening * np.pi / 180)
    if len(earth) == 2:
        ax.fill(earth[0], earth[1], "b", alpha=0.2, label="EARTH")
    else:
        ax.fill(earth[0], earth[1], "b", alpha=0.2, label="EARTH")
        ax.fill(earth[2], earth[3], "b", alpha=0.2)

    # Plot GRB contours from fit
    # Get contours
    (
        x_contour,
        y_contour,
        val_contour,
        x_contour_1,
        x_contour_2,
        val_contour_1,
        val_contour_2,
    ) = get_contours(model, post_equal_weights_file)

    if len(x_contour_1) > 0:
        ax.contourf(
            x_contour_1,
            y_contour,
            val_contour_1,
            levels=[0, 0.68268949, 0.9545],
            colors=["navy", "lightgreen"],
        )
    if len(x_contour_2) > 0:
        ax.contourf(
            x_contour_2,
            y_contour,
            val_contour_2,
            levels=[0, 0.68268949, 0.9545],
            colors=["navy", "lightgreen"],
        )

    # Plot GRB best fit
    ra_center = ra * np.pi / 180
    dec_center = dec * np.pi / 180
    if ra_center > np.pi:
        ra_center = ra_center - 2 * np.pi

    ax.scatter(
        ra_center, dec_center, label="Balrog Position", s=40, color="green", marker="*"
    )
    ax.annotate(
        f"Balrog Position {grb_name}",
        xy=(ra_center, dec_center),  # theta, radius
        xytext=(0.55, 0.15),  # fraction, fraction
        textcoords="figure fraction",
        arrowprops=dict(
            facecolor="black", shrink=0.02, width=1, headwidth=5, headlength=5
        ),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    #different markers for better visualization
    markers=["o","X","^"]

    # Plot Bright Sources
    for m, (name, dictionary) in enumerate(bright_sources.items()):
        ra = np.deg2rad(dictionary["ra"])
        dec = np.deg2rad(dictionary["dec"])
        if ra>np.pi:
            ra-=2*np.pi
        ax.scatter(ra, dec, label=name, s=30, color="red",marker=markers[m])

    # Plot SGRs
    for m, (name, dictionary) in enumerate(SGRs.items()):
        ra = np.deg2rad(dictionary["ra"])
        dec = np.deg2rad(dictionary["dec"])
        if ra>np.pi:
            ra-=2*np.pi
        ax.scatter(ra, dec, label=name, s=30, color="orange",marker=markers[m])

    # if we have a swift position plot it here
    if swift is not None:
        # Plot SWIFT position if there is one
        ra_swift = float(swift["ra"]) * np.pi / 180
        dec_swift = float(swift["dec"]) * np.pi / 180
        if ra_swift > np.pi:
            ra_swift = ra_swift - 2 * np.pi
        ax.scatter(
            ra_swift,
            dec_swift,
            label="SWIFT Position",
            s=40,
            marker="X",
            color="magenta",
            alpha=0.2,
        )
        ax.annotate(
            "SWIFT Position SWIFT-trigger {}".format(swift["trigger"]),
            xy=(ra_swift, dec_swift),  # theta, radius
            xytext=(0.55, 0.78),  # fraction, fraction
            textcoords="figure fraction",
            arrowprops=dict(
                facecolor="black", shrink=0.02, width=1, headwidth=5, headlength=5
            ),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # set title, legend and grid
    plt.title(f"{grb_name} Bright Objects (J2000)", y=1.08)
    ax.grid()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

    # save figure
    file_utils.if_dir_containing_file_not_existing_then_make(save_path)
    fig.savefig(save_path, bbox_inches="tight", dpi=500)


def azimuthal_plot_sat_frame(grb_name, trigdat_file, ra, dec, save_path):
    """
    plot azimuth plot in sat frame to check if burst comes from the solar panel sides
    :return:
    """
    ra_center = ra * np.pi / 180
    dec_center = dec * np.pi / 180
    if ra_center > np.pi:
        ra_center = ra_center - 2 * np.pi

    with fits.open(trigdat_file) as f:
        quat = f["TRIGRATE"].data["SCATTITD"][0]
        sc_pos = f["TRIGRATE"].data["EIC"][0]
        times = f["TRIGRATE"].data["TIME"][0]

    cone_opening = 45.0  # cone opening for solar panel side in deg
    loc_icrs = SkyCoord(
        ra=ra_center * 180 / np.pi,
        dec=dec_center * 180 / np.pi,
        unit="deg",
        frame="icrs",
    )
    q1, q2, q3, q4 = quat
    scx, scy, scz = sc_pos
    loc_sat = loc_icrs.transform_to(
        GBMFrame(
            quaternion_1=q1,
            quaternion_2=q2,
            quaternion_3=q3,
            quaternion_4=q4,
            sc_pos_X=scx,
            sc_pos_Y=scy,
            sc_pos_Z=scz,
        )
    )
    ra_sat = Angle(loc_sat.lon.deg * unit.degree)
    dec_sat = Angle(loc_sat.lat.deg * unit.degree)
    ra_sat.wrap_at("180d", inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    # Fill area where the solar panels may cause a systematic error

    r_bound = np.arange(0, 200, 0.5)
    phi_bound = np.ones_like(r_bound)
    ax.fill_betweenx(
        r_bound,
        phi_bound * (np.pi / 2 - cone_opening * (np.pi / 180)),
        phi_bound * (np.pi / 2 + cone_opening * (np.pi / 180)),
        color="grey",
        alpha=0.2,
        label="solar panel sides",
    )
    ax.fill_betweenx(
        r_bound,
        phi_bound * (-np.pi / 2 - cone_opening * (np.pi / 180)),
        phi_bound * (-np.pi / 2 + cone_opening * (np.pi / 180)),
        color="grey",
        alpha=0.2,
    )

    # Fill other area and label with b0 and b1 side
    ax.fill_betweenx(
        r_bound,
        phi_bound * (-np.pi / 2 + cone_opening * (np.pi / 180)),
        phi_bound * (np.pi / 2 - cone_opening * (np.pi / 180)),
        color="lime",
        alpha=0.2,
        label="b0 side",
    )
    ax.fill_betweenx(
        r_bound,
        phi_bound * (-np.pi / 2 - cone_opening * (np.pi / 180)),
        phi_bound * (np.pi / 2 + cone_opening * (np.pi / 180)),
        color="blue",
        alpha=0.2,
        label="b1 side",
    )

    # SAT coordinate system#
    ax.quiver(np.pi / 2, 0, 0, 1, scale=2.0)
    ax.text((np.pi / 2) * 1.07, 0.9, "y")
    ax.quiver(0, 0, 1, 0, scale=2.0)
    ax.text(-(np.pi / 2) * 0.07, 0.9, "x")
    ax.set_rlim((0, 1))
    ax.set_yticklabels([])

    # Plot Burst direction in Sat-Coord#
    phi_b = ra_sat.value * np.pi / 180
    u = np.cos(phi_b)
    v = np.sin(phi_b)

    q = ax.quiver(0, 0, u, v, scale=2.0, color="yellow", linewidth=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.8))
    ax.quiverkey(q, X=1.3, Y=0.5, U=0.4, label=f"{grb_name}", labelpos="N")
    ax.set_title(f"{grb_name} direction in the sat. frame", y=1.08)

    file_utils.if_dir_containing_file_not_existing_then_make(save_path)

    fig.savefig(save_path, bbox_inches="tight", dpi=1000)


def swift_gbm_plot(
    grb_name, ra, dec, model, post_equal_weights_file, save_path, swift=None
):
    """
    If swift postion known make a small area plot with grb position, error contours and Swift position (in deg)
    This Plot has to be made AFTER the mollweide plot.
    :return:
    """
    if swift is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ra_center = ra * np.pi / 180
        dec_center = dec * np.pi / 180
        if ra_center > np.pi:
            ra_center = ra_center - 2 * np.pi

        # Get contours
        (
            x_contour,
            y_contour,
            val_contour,
            x_contour_1,
            x_contour_2,
            val_contour_1,
            val_contour_2,
        ) = get_contours(model, post_equal_weights_file)

        x_contour_1 = x_contour[x_contour < np.pi]
        x_contour_2 = x_contour[x_contour > np.pi] - 2 * np.pi

        val_contour_1 = val_contour[:, x_contour < np.pi]
        val_contour_2 = val_contour[:, x_contour > np.pi]

        if swift["ra"] > 180:
            swift_ra = float(swift["ra"]) - 360
        else:
            swift_ra = float(swift["ra"])
        # plot Balrog position with errors and swift position
        if len(x_contour_1):
            ax.contourf(
                x_contour_1 * 180 / np.pi,
                y_contour * 180 / np.pi,
                val_contour_1,
                levels=[0, 0.68268949, 0.9545],
                colors=["navy", "lightgreen"],
            )
        if len(x_contour_2):
            ax.contourf(
                x_contour_2 * 180 / np.pi,
                y_contour * 180 / np.pi,
                val_contour_2,
                levels=[0, 0.68268949, 0.9545],
                colors=["navy", "lightgreen"],
            )
        ax.scatter(
            swift_ra,
            swift["dec"],
            label="SWIFT Position",
            s=40,
            marker="X",
            color="magenta",
            alpha=0.5,
        )
        ax.scatter(
            ra_center * 180 / np.pi,
            dec_center * 180 / np.pi,
            label="Balrog Position",
            s=40,
            marker="*",
            color="green",
            alpha=0.5,
        )
        ra_diff = np.abs(ra_center - swift_ra * np.pi / 180)
        dec_diff = np.abs(dec_center - swift["dec"] * np.pi / 180)
        print(ra_diff)
        print(dec_diff)
        # choose a decent plotting range
        if ra_diff * 180 / np.pi > 2:
            ax.set_xlim(
                (
                    ra_center * 180 / np.pi - (1.1) * ra_diff * 180 / np.pi,
                    ra_center * 180 / np.pi + (1.1) * ra_diff * 180 / np.pi,
                )
            )
        else:
            ax.set_xlim((ra_center * 180 / np.pi - 2, ra_center * 180 / np.pi + 2))
        if dec_diff * 180 / np.pi > 2:
            ax.set_ylim(
                (
                    dec_center * 180 / np.pi - (1.1) * dec_diff * 180 / np.pi,
                    dec_center * 180 / np.pi + (1.1) * dec_diff * 180 / np.pi,
                )
            )
        else:
            ax.set_ylim((dec_center * 180 / np.pi - 2, dec_center * 180 / np.pi + 2))
        # plot error contours
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("DEC (deg)")
        plt.title(f"{grb_name} Position (J2000)", y=1.08)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})
        ax.grid(True)

        # save plot
        file_utils.if_dir_containing_file_not_existing_then_make(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=1000)


def interactive_3D_plot(
    post_equal_weights_file, trigdat_file, used_dets, model, save_path
):
    # Plot 10 degree grid
    trace_grid = []
    phi_l = np.arange(-180, 181, 10)  # file size!#
    theta_l = np.arange(-90, 91, 10)  # file size!#
    scale_factor_grid = 1.02
    b_side_angle = 45  # has to be <90
    for phi in phi_l:
        x, y, z = xyz(phi, theta_l)
        trace_grid.append(
            go.Scatter3d(
                x=scale_factor_grid * x,
                y=scale_factor_grid * y,
                z=scale_factor_grid * z,
                legendgroup="group",
                showlegend=False,
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                hoverinfo=None,
            )
        )

    for theta in theta_l:
        theta_m = np.ones_like(phi_l) * theta
        x, y, z = xyz(phi_l, theta_m)
        trace_grid.append(
            go.Scatter3d(
                x=scale_factor_grid * x,
                y=scale_factor_grid * y,
                z=scale_factor_grid * z,
                legendgroup="group",
                showlegend=False,
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                hoverinfo=None,
            )
        )
    # equator
    theta_m = np.ones_like(phi_l) * 0
    x, y, z = xyz(phi_l, theta_m)
    trace_grid.append(
        go.Scatter3d(
            x=np.array(scale_factor_grid * x),
            y=np.array(scale_factor_grid * y),
            z=np.array(scale_factor_grid * z),
            legendgroup="group",
            name="Spherical Grid (10 deg steps)",
            mode="lines",
            line=dict(color="black", width=3),
            hoverinfo=None,
        )
    )

    # PLOT B0 and B1 Side and Solar panel
    phi = np.concatenate(
        [
            np.arange(0, b_side_angle - 1, (b_side_angle - 1) / 10),
            np.arange(b_side_angle - 1, b_side_angle + 1, 0.1),
            np.arange(b_side_angle + 1, 180 - (b_side_angle + 1), 2 * b_side_angle / 4),
            np.arange(180 - (b_side_angle + 1), 180 - (b_side_angle - 1), 0.1),
            np.arange(180 - (b_side_angle - 1), 181, (b_side_angle - 1.0) / 10.0),
        ]
    )  # file size!#
    phi = np.concatenate([phi, -np.flip(phi[:-1], 0)])
    theta = np.arange(-90, 91, 5)  # file size!#
    # phi, theta = np.mgrid[-180:180:720j, -90:90:18j]
    phi, theta = np.meshgrid(phi, theta)
    x, y, z = xyz(phi, theta)
    points = np.array([x, y, z])

    b0_zen = 0
    b0_azi = 0
    b1_zen = 0
    b1_azi = 180
    b0_x, b0_y, b0_z = xyz(b0_azi, b0_zen)
    b1_x, b1_y, b1_z = xyz(b1_azi, b1_zen)

    idx_b0 = np.abs(phi) < b_side_angle
    idx_b1 = np.abs(phi) > 180 - b_side_angle
    xin_b0, yin_b0, zin_b0 = xyz(phi, theta)
    xin_b1, yin_b1, zin_b1 = xyz(phi, theta)
    xin_s, yin_s, zin_s = xyz(phi, theta)

    xin_b0[~idx_b0] = np.nan
    yin_b0[~idx_b0] = np.nan
    zin_b0[~idx_b0] = np.nan

    xin_b1[~idx_b1] = np.nan
    yin_b1[~idx_b1] = np.nan
    zin_b1[~idx_b1] = np.nan

    xin_s[idx_b1] = np.nan
    yin_s[idx_b1] = np.nan
    zin_s[idx_b1] = np.nan
    xin_s[idx_b0] = np.nan
    yin_s[idx_b0] = np.nan
    zin_s[idx_b0] = np.nan

    contours = go.surface.Contours(
        x=go.surface.contours.X(highlight=False),
        y=go.surface.contours.Y(highlight=False),
        z=go.surface.contours.Z(highlight=False),
    )

    theta = np.arcsin(z) * 180 / np.pi
    phi = np.arctan2(x, y) * 180 / np.pi
    my_text = []
    for i in range(len(phi)):
        te = []
        for j in range(len(phi[0])):
            te.append("phi:{}<br>theta:{}".format(phi[i, j], theta[i, j]))
        my_text.append(te)
    my_text = np.array(my_text)
    colorscale_b0 = [[0, "rgb(117,201,196)"], [1, "rgb(117,201,196)"]]
    trace_b0 = go.Surface(
        x=xin_b0,
        y=yin_b0,
        z=zin_b0,
        name="b0-side",
        showscale=False,
        colorscale=colorscale_b0,
        surfacecolor=np.ones_like(z),
        opacity=1,
        contours=contours,
        text=my_text,
        hoverinfo="text+name",
    )
    colorscale_b1 = [[0, "rgb(201,117,117)"], [1, "rgb(201,117,117)"]]
    trace_b1 = go.Surface(
        x=xin_b1,
        y=yin_b1,
        z=zin_b1,
        name="b1-side",
        showscale=False,
        colorscale=colorscale_b1,
        surfacecolor=np.ones_like(z),
        opacity=1,
        contours=contours,
        text=my_text,
        hoverinfo="text+name",
    )
    colorscale_s = [[0, "grey"], [1, "grey"]]
    trace_s = go.Surface(
        x=xin_s,
        y=yin_s,
        z=zin_s,
        name="solar_panel side",
        showscale=False,
        colorscale=colorscale_s,
        surfacecolor=np.ones_like(z),
        opacity=1,
        contours=contours,
        text=my_text,
        hoverinfo="text+name",
    )

    # PLOT DETS - dets in list used dets will be plotted solid all other dashed
    trace_dets = []
    color_dict = {
        "n0": "blue",
        "n1": "navy",
        "n2": "crimson",
        "n3": "lightgreen",
        "n4": "orchid",
        "n5": "brown",
        "n6": "firebrick",
        "n7": "plum",
        "n8": "darkgreen",
        "n9": "olive",
        "na": "aqua",
        "nb": "darkorange",
        "b0": "darkmagenta",
        "b1": "indigo",
    }
    det_pointing = {
        "n0": [45.9, 90 - 20.6],
        "n1": [45.1, 90 - 45.3],
        "n2": [58.4, 90 - 90.2],
        "n3": [314.9, 90 - 45.2],
        "n4": [303.2, 90 - 90.3],
        "n5": [3.4, 90 - 89.8],
        "n6": [224.9, 90 - 20.4],
        "n7": [224.6, 90 - 46.2],
        "n8": [236.6, 90 - 90],
        "n9": [135.2, 90 - 45.6],
        "na": [123.7, 90 - 90.4],
        "nb": [183.7, 90 - 90.3],
        "b0": [0.01, 90 - 90.01],
        "b1": [180.01, 90 - 90.01],
    }
    for keys in det_pointing:
        det_opening = 40  # in deg
        pointing = det_pointing[keys]
        ra_d = pointing[0] * np.pi / 180
        dec_d = pointing[1] * np.pi / 180
        scale_factor_d = 1.01
        theta_l = np.linspace(-np.pi / 2, np.pi / 2, 720)  # file size!#
        phi_res_0 = []
        phi_res_1 = []
        for theta in theta_l:
            phi_res_0.append(phi_0(theta, ra_d, dec_d, det_opening * np.pi / 180))
            phi_res_1.append(phi_1(theta, ra_d, dec_d, det_opening * np.pi / 180))

        phi_res_0 = np.array(phi_res_0)
        phi_res_1 = np.array(phi_res_1)
        theta_all = np.concatenate([theta_l, np.flip(theta_l, 0)])
        phi_all = np.concatenate([phi_res_0, np.flip(phi_res_1, 0)])
        mask = phi_all < 100
        theta_all = theta_all[mask]
        phi_all = phi_all[mask]
        theta_all = np.concatenate([theta_all, theta_all[0:1]])
        phi_all = np.concatenate([phi_all, phi_all[:1]])
        x = np.cos(theta_all) * np.cos(phi_all)
        y = np.cos(theta_all) * np.sin(phi_all)
        z = np.sin(theta_all)
        # plot earth
        name = str(keys)
        color = str(color_dict[keys])
        if name in used_dets:
            trace_dets.append(
                go.Scatter3d(
                    x=scale_factor_d * x,
                    y=scale_factor_d * y,
                    z=scale_factor_d * z,
                    name=name,
                    legendgroup="used detectors",
                    mode="lines",
                    line=dict(color=color, width=5, dash="solid"),
                    hoverinfo="name",
                )
            )
        else:
            trace_dets.append(
                go.Scatter3d(
                    x=scale_factor_d * x,
                    y=scale_factor_d * y,
                    z=scale_factor_d * z,
                    name=name,
                    mode="lines",
                    legendgroup="unused detectors",
                    line=dict(color=color, width=5, dash="dash"),
                    hoverinfo="name",
                )
            )

    with fits.open(trigdat_file) as f:
        quat = f["TRIGRATE"].data["SCATTITD"][0]
        sc_pos = f["TRIGRATE"].data["EIC"][0]
        times = f["TRIGRATE"].data["TIME"][0]

    # Plot Earth Shadow
    det = gbm_detector_list["n0"](
        quaternion=quat, sc_pos=sc_pos, time=astro_time.Time(utc(times))
    )
    earth_pos_sat = det.earth_position
    ra_earth_sat = earth_pos_sat.lon.deg
    dec_earth_sat = earth_pos_sat.lat.deg
    # earth_pos
    xe, ye, ze = xyz(ra_earth_sat, dec_earth_sat)
    earth_vec = np.array([xe, ye, ze])
    opening_angle = 67
    # points on sphere
    theta_l = np.concatenate(
        [
            np.linspace(-np.pi / 2, -np.pi / 2 + 0.1, 30),
            np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 400),
            np.linspace(np.pi / 2 - 0.1, np.pi / 2, 30),
        ]
    )  # file size!#
    theta_final = []
    phi_final = []
    phi_l = np.arange(-np.pi, np.pi + 0.1, 0.1)
    for theta in theta_l:
        for phi in phi_l:
            x, y, z = xyz(phi * 180 / np.pi, theta * 180 / np.pi)
            angle = np.arccos(np.dot(np.array([x, y, z]), earth_vec))
            if angle < opening_angle * np.pi / 180:
                theta_final.append(theta)
                phi_final.append(phi)
    theta_final = np.array(theta_final)
    phi_final = np.array(phi_final)

    x = np.cos(theta_final) * np.cos(phi_final)
    y = np.cos(theta_final) * np.sin(phi_final)
    z = np.sin(theta_final)
    scale_factor_earth = 1.005
    colorscale = [[0, "navy"], [1, "navy"]]

    theta = np.arcsin(z) * 180 / np.pi
    phi = np.arctan2(x, y) * 180 / np.pi
    my_text = []
    for i in range(len(phi)):
        my_text.append("phi:{}<br>theta:{}".format(phi[i], theta[i]))
    my_text = np.array(my_text)
    trace_earth = go.Mesh3d(
        x=scale_factor_earth * x,
        y=scale_factor_earth * y,
        z=scale_factor_earth * z,
        showscale=False,
        name="earth",
        color="navy",
        alphahull=0,
        text=my_text,
        hoverinfo="text+name",
    )

    # Plot Balrog ERROR CONTOURS
    # Load data from chain with chain consumer
    chain = loadtxt2d(post_equal_weights_file)

    # Get parameter for model
    parameter = model_param_lookup[model]

    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(
        contour_labels="sigma", colors="#cd5c5c", label_font_size=20
    )
    # ra_contour, dec_contour, val_contour = c1.plotter.get_contours_list('ra', 'dec')  # ra, dec in deg here
    chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(
        None, None, None, None, color_p=True, blind=None
    )
    hist, ra_contour, dec_contour = c1.plotter._get_smoothed_histogram2d(
        chains[0], "ra (deg)", "dec (deg)"
    )  # ra, dec in deg here
    hist[hist == 0] = 1e-16
    val_contour = c1.plotter._convert_to_stdev(hist.T)
    ra_con, dec_con = np.meshgrid(ra_contour, dec_contour)
    a = np.array([ra_con, dec_con]).T
    res = []
    q1, q2, q3, q4 = quat
    scx, scy, scz = sc_pos
    for a_inter in a:
        loc_icrs = SkyCoord(
            ra=a_inter[:, 0], dec=a_inter[:, 1], unit="deg", frame="icrs"
        )
        loc_sat = loc_icrs.transform_to(
            GBMFrame(
                quaternion_1=q1,
                quaternion_2=q2,
                quaternion_3=q3,
                quaternion_4=q4,
                sc_pos_X=scx,
                sc_pos_Y=scy,
                sc_pos_Z=scz,
            )
        )
        ra_sat = Angle(loc_sat.lon.deg * unit.degree).value
        dec_sat = Angle(loc_sat.lat.deg * unit.degree).value
        res.append(np.stack((ra_sat, dec_sat), axis=-1))
    res = np.array(res)
    scale_factor_con = 1.02
    x, y, z = xyz(res[:, :, 0], res[:, :, 1])
    x = scale_factor_con * x
    y = y * scale_factor_con
    z = z * scale_factor_con
    colorscale = [[0, "green"], [1.0 / 3.0, "orange"], [2.0 / 3.0, "red"], [1, "grey"]]
    conf_levels = [0.68, 0.95, 0.99]
    trace_conf_l = []
    theta = np.arcsin(z) * 180 / np.pi
    phi = np.arctan2(x, y) * 180 / np.pi
    my_text = []
    for i in range(len(phi)):
        te = []
        for j in range(len(phi[0])):
            te.append("phi:{}<br>theta:{}".format(phi[i, j], theta[i, j]))
        my_text.append(te)
    my_text = np.array(my_text)
    for conf in conf_levels:
        x2n, y2n, z2n = (
            np.where(val_contour < conf, x, None),
            np.where(val_contour < conf, y, None),
            np.where(val_contour < conf, z, None),
        )
        trace_conf = go.Surface(
            x=x2n,
            y=y2n,
            z=z2n,
            cmin=0,
            cmax=3,
            showscale=False,
            colorscale=colorscale,
            surfacecolor=z2n,
            name="Balrog {} confidence level".format(conf),
            text=my_text,
            hoverinfo="text+name",
        )
        lx = len(trace_conf["z"])
        ly = len(trace_conf["z"][0])
        out = []
        x_sigma1 = []
        for i in range(lx):
            temp = []
            for j in range(ly):
                if val_contour[i, j] < 0.68:
                    temp.append(0)
                elif val_contour[i, j] < 0.95:
                    temp.append(1)
                elif val_contour[i, j] < 0.99:
                    temp.append(2)
                else:
                    temp.append(3)
            out.append(temp)
        # PLOT BESTFIT and SWIFT (if given)
        trace_conf["surfacecolor"] = out
        trace_conf_l.append(trace_conf)

    # TODO add swift position

    # add data together
    data = (
        [trace_b0, trace_b1, trace_s, trace_earth]
        + trace_grid
        + trace_dets
        + trace_conf_l
    )
    # change layout
    layout = go.Layout(
        dict(
            hovermode="closest",
            autosize=True,
            # width=1000,
            height=800,
            scene=dict(
                xaxis=dict(
                    title="",
                    autorange=True,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    showspikes=False,
                ),
                yaxis=dict(
                    title="",
                    autorange=True,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    showspikes=False,
                ),
                zaxis=dict(
                    title="",
                    autorange=True,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    showspikes=False,
                ),
            ),
        )
    )
    # create figure
    fig = go.Figure(data=data, layout=layout)

    output = plotly.offline.plot(
        fig, auto_open=False, output_type="div", include_plotlyjs=False, show_link=False
    )

    file_utils.if_dir_containing_file_not_existing_then_make(save_path)
    with open(save_path, "w") as text_file:
        text_file.write(output)

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
    sep = np.arccos(
        c1 * np.cos(theta) * np.cos(phi)
        + c2 * np.cos(theta) * np.sin(phi)
        + c3 * np.sin(theta)
    )
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

        phi_circle_0 = np.concatenate(
            (phi_circle_0, np.flip(phi_circle_0, 0), np.array([phi_circle_0[0]]))
        )
        phi_circle_1 = np.concatenate(
            (phi_circle_1, np.flip(phi_circle_1, 0), np.array([phi_circle_1[0]]))
        )
        theta_circle_0 = np.concatenate(
            (theta_min_0, np.flip(theta_max_0, 0), np.array([theta_min_0[0]]))
        )
        theta_circle_1 = np.concatenate(
            (theta_min_1, np.flip(theta_max_1, 0), np.array([theta_min_1[0]]))
        )
        return [phi_circle_0, theta_circle_0, phi_circle_1, theta_circle_1]
    else:
        phi_circle = np.concatenate(
            (phi_circle, np.flip(phi_circle, 0), np.array([phi_circle[0]]))
        )
        theta_circle = np.concatenate(
            (theta_min, np.flip(theta_max, 0), np.array([theta_min[0]]))
        )
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
    z = (
        -c
        + (a ** 2 * c / (a ** 2 + b ** 2))
        + d
        - (a ** 2 * d / (a ** 2 + b ** 2))
        + (a * np.sqrt(-(b ** 2) * (-(a ** 2) - b ** 2 + c ** 2 - 2 * c * d + d ** 2)))
        / (a ** 2 + b ** 2)
    ) / b
    n = (
        -a * c
        + a * d
        - np.sqrt(
            a ** 2 * b ** 2
            + b ** 4
            - b ** 2 * c ** 2
            + 2 * b ** 2 * c * d
            - b ** 2 * d ** 2
        )
    ) / (a ** 2 + b ** 2)
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
    z = (
        -c
        + (a ** 2 * c / (a ** 2 + b ** 2))
        + d
        - (a ** 2 * d / (a ** 2 + b ** 2))
        - (a * np.sqrt(-(b ** 2) * (-(a ** 2) - b ** 2 + c ** 2 - 2 * c * d + d ** 2)))
        / (a ** 2 + b ** 2)
    ) / b
    n = (
        -a * c
        + a * d
        + np.sqrt(
            a ** 2 * b ** 2
            + b ** 4
            - b ** 2 * c ** 2
            + 2 * b ** 2 * c * d
            - b ** 2 * d ** 2
        )
    ) / (a ** 2 + b ** 2)
    phi = np.arctan2(z, n)
    return phi


def get_contours(model, post_equal_weigts_file):
    # Get parameter for model
    parameter = model_param_lookup[model]

    # get contours
    chain = loadtxt2d(post_equal_weigts_file)

    c1 = ChainConsumer()
    c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(
        plot_hists=False, contour_labels="sigma", colors="#cd5c5c", flip=False
    )

    # x_contour, y_contour, val_contour = c1.plotter.get_contours_list('ra (deg)', 'dec (deg)')

    chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(
        None, None, None, None, color_p=True, blind=None
    )
    hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(
        chains[0], "ra (deg)", "dec (deg)"
    )  # ra, dec in deg here
    hist[hist == 0] = 1e-16
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
        c1.add_chain(chain[:, :-1][:, :2], parameters=parameter[:2]).configure(
            plot_hists=False, contour_labels="sigma", colors="#cd5c5c", flip=False
        )

        # x_contour, y_contour, val_contour = c1.plotter.get_contours_list('ra (deg)', 'dec (deg)')

        chains, parameters, truth, extents, blind, log_scales = c1.plotter._sanitise(
            None, None, None, None, color_p=True, blind=None
        )
        hist, x_contour, y_contour = c1.plotter._get_smoothed_histogram2d(
            chains[0], "ra (deg)", "dec (deg)"
        )  # ra, dec in deg here
        hist[hist == 0] = 1e-16
        val_contour = c1.plotter._convert_to_stdev(hist.T)

    x_contour = x_contour * np.pi / 180
    y_contour = y_contour * np.pi / 180

    # split in ra area between 0 and pi & pi and 2 pi and wrap the second one to -pi to 0
    x_contour_1 = x_contour[x_contour < np.pi]
    x_contour_2 = x_contour[x_contour > np.pi] - 2 * np.pi

    val_contour_1 = val_contour[:, x_contour < np.pi]
    val_contour_2 = val_contour[:, x_contour > np.pi]

    return (
        x_contour,
        y_contour,
        val_contour,
        x_contour_1,
        x_contour_2,
        val_contour_1,
        val_contour_2,
    )
