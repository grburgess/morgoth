from datetime import datetime
import pytz
import urllib

import astropy.io.fits as fits
import numpy as np
import yaml
from chainconsumer import ChainConsumer
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord

from morgoth.exceptions.custom_exceptions import *
from morgoth.utils.env import get_env_value

from morgoth.utils.swift_check import check_swift

from gbmgeometry.gbm_frame import GBMFrame
from astropy.coordinates import SkyCoord
import astropy.units as unit

from astropy.table import Table
from astropy.coordinates import Angle
import pandas as pd
import os, ssl
import requests


base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class ResultReader(object):
    def __init__(
        self,
        grb_name,
        report_type,
        version,
        trigger_file,
        time_selection_file,
        background_file,
        post_equal_weights_file,
        result_file,
        trigdat_file,
    ):
        self.grb_name = grb_name
        self.report_type = report_type
        self.version = version

        self._K = None
        self._K_err = None
        self._index = None
        self._index_err = None
        self._xc = None
        self._xc_err = None
        self._alpha = None
        self._alpha_err = None
        self._xp = None
        self._xp_err = None
        self._beta = None
        self._beta_err = None

        # read trigger output
        self._read_trigger(trigger_file)

        # read time selection values
        self._read_time_selection(time_selection_file)

        # read background file
        self._read_background_fit(background_file)

        # read parameter values
        self._read_fit_result(result_file)

        # read parameter values
        self._read_post_equal_weights_file(post_equal_weights_file)

        # Read trigdat file
        self._read_trigdat_file(trigdat_file)

        # Check the GCN Archive for the GRB letter
        self._grb_name_gcn = check_letter(
            trigger_number=self._trigger_number, grb_name=self.grb_name
        )

        # Check catalog of bright gamma sources and get separation to GRB position
        self._sep_bright_sources()

        # Check catalog of SGRs and get separation to GRB position
        self._sep_SGRs()

        # Create a report containing all the results of the pipeline
        self._build_report()

    def _read_trigdat_file(self, trigdat_file):
        """
        Read trigdat file for sc_pos and quats, needed to calculate GRB position in sat frame
        :param trigdat_file: Path to trigdat data file
        :return:
        """
        ra_center = self._ra * np.pi / 180
        dec_center = self._dec * np.pi / 180
        if ra_center > np.pi:
            ra_center = ra_center - 2 * np.pi

        if type(trigdat_file)==str:
            with fits.open(trigdat_file) as f:
                quat = f["TRIGRATE"].data["SCATTITD"][0]
                sc_pos = f["TRIGRATE"].data["EIC"][0]
                times = f["TRIGRATE"].data["TIME"][0]

                data_timestamp_goddard = f["PRIMARY"].header["DATE"] + ".000Z"
        else:
            with fits.open(trigdat_file.path) as f:
                quat = f["TRIGRATE"].data["SCATTITD"][0]
                sc_pos = f["TRIGRATE"].data["EIC"][0]
                times = f["TRIGRATE"].data["TIME"][0]

                data_timestamp_goddard = f["PRIMARY"].header["DATE"] + ".000Z"

        datetime_ob_goddard = pytz.timezone("US/Eastern").localize(
            datetime.strptime(data_timestamp_goddard, "%Y-%m-%dT%H:%M:%S.%fZ")
        )

        # datetime_ob_utc = datetime_ob_goddard.astimezone(pytz.timezone("UTC"))

        # self._data_timestamp = datetime_ob_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        self._data_timestamp = datetime_ob_goddard.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

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

        phi_sat = Angle(loc_sat.lon.deg * unit.degree)
        theta_sat = Angle(loc_sat.lat.deg * unit.degree)
        phi_sat.wrap_at("180d", inplace=True)

        self._phi_sat = phi_sat.value
        self._theta_sat = theta_sat.value

    def _read_fit_result(self, result_file):

        with fits.open(result_file) as f:
            values = f["ANALYSIS_RESULTS"].data["VALUE"]
            pos_error = f["ANALYSIS_RESULTS"].data["POSITIVE_ERROR"]
            neg_error = f["ANALYSIS_RESULTS"].data["NEGATIVE_ERROR"]

        self._ra = values[0]
        self._ra_pos_err = pos_error[0]
        self._ra_neg_err = neg_error[0]

        if np.absolute(self._ra_pos_err) > np.absolute(self._ra_neg_err):
            self._ra_err = np.absolute(self._ra_pos_err)
        else:
            self._ra_err = np.absolute(self._ra_neg_err)

        self._dec = values[1]
        self._dec_pos_err = pos_error[1]
        self._dec_neg_err = neg_error[1]

        if np.absolute(self._dec_pos_err) > np.absolute(self._dec_neg_err):
            self._dec_err = np.absolute(self._dec_pos_err)
        else:
            self._dec_err = np.absolute(self._dec_neg_err)

        if self.report_type == "trigdat":
            self._K = values[2]
            self._K_pos_err = pos_error[2]
            self._K_neg_err = neg_error[2]

            if np.absolute(self._K_pos_err) > np.absolute(self._K_neg_err):
                self._K_err = np.absolute(self._K_pos_err)
            else:
                self._K_err = np.absolute(self._K_neg_err)

            self._index = values[3]
            self._index_pos_err = pos_error[3]
            self._index_neg_err = neg_error[3]

            if np.absolute(self._index_pos_err) > np.absolute(self._index_neg_err):
                self._index_err = np.absolute(self._index_pos_err)
            else:
                self._index_err = np.absolute(self._index_neg_err)

            try:
                self._xc = values[4]
                self._xc_pos_err = pos_error[4]
                self._xc_neg_err = neg_error[4]
                if np.absolute(self._xc_pos_err) > np.absolute(self._xc_neg_err):
                    self._xc_err = np.absolute(self._xc_pos_err)
                else:
                    self._xc_err = np.absolute(self._xc_neg_err)
                self._model = "cpl"
            except:
                self._model = "pl"

        elif self.report_type == "tte":
            self._model = "band"
            self._K = values[2]
            self._K_pos_err = pos_error[2]
            self._K_neg_err = neg_error[2]

            if np.absolute(self._K_pos_err) > np.absolute(self._K_neg_err):
                self._K_err = np.absolute(self._K_pos_err)
            else:
                self._K_err = np.absolute(self._K_neg_err)

            self._alpha = values[3]
            self._alpha_pos_err = pos_error[3]
            self._alpha_neg_err = neg_error[3]

            if np.absolute(self._alpha_pos_err) > np.absolute(self._alpha_neg_err):
                self._alpha_err = np.absolute(self._alpha_pos_err)
            else:
                self._alpha_err = np.absolute(self._alpha_neg_err)

            self._xp = values[4]
            self._xp_pos_err = pos_error[4]
            self._xp_neg_err = neg_error[4]

            if np.absolute(self._xp_pos_err) > np.absolute(self._xp_neg_err):
                self._xp_err = np.absolute(self._xp_pos_err)
            else:
                self._xp_err = np.absolute(self._xp_neg_err)

            self._beta = values[5]
            self._beta_pos_err = pos_error[5]
            self._beta_neg_err = neg_error[5]

            if np.absolute(self._beta_pos_err) > np.absolute(self._beta_neg_err):
                self._beta_err = np.absolute(self._beta_pos_err)
            else:
                self._beta_err = np.absolute(self._beta_neg_err)

        else:
            raise UnkownReportType(
                "The specified report type is not valid. Valid report types: (trigdat, tte)"
            )

    def _read_trigger(self, trigger_file):
        with open(trigger_file, "r") as f:
            data = yaml.safe_load(f)

            self._trigger_number = data["trigger_number"]
            self._trigger_timestamp = data["trigger_time"]

            self._most_likely = f"{data['most_likely']} {data['most_likely_prob']}%"
            self._second_most_likely = (
                f"{data['most_likely_2']} {data['most_likely_prob_2']}%"
            )

            self._swift = check_swift(
                datetime.strptime(self._trigger_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            )
            # self._swift = {"ra": convert_to_float(swift[ra]),
            #               "dec": convert_to_float(swift[dec]),
            #               "trigger": int(swift["trigger"])}

    def _read_time_selection(self, time_selection_file):
        with open(time_selection_file, "r") as f:
            data = yaml.safe_load(f)

            self._bkg_neg_start = data["background_time"]["before"]["start"]
            self._bkg_neg_stop = data["background_time"]["before"]["stop"]
            self._bkg_pos_start = data["background_time"]["after"]["start"]
            self._bkg_pos_stop = data["background_time"]["after"]["stop"]
            self._active_time_start = data["active_time"]["start"]
            self._active_time_stop = data["active_time"]["stop"]
            self._poly_order = data["poly_order"]

    def _read_background_fit(self, background_fit_file):
        with open(background_fit_file, "r") as f:
            data = yaml.safe_load(f)
            self._used_detectors = data["use_dets"]

    def _read_post_equal_weights_file(self, post_equal_weights_file):

        # Sometimes chainconsumer does not give an error - In this case we will need the errors from the
        # 3ml fits files
        (
            self._ra,
            ra_err,
            self._dec,
            dec_err,
            self._balrog_one_sig_err_circle,
            self._balrog_two_sig_err_circle,
        ) = get_best_fit_with_errors(post_equal_weights_file, self._model)

        if ra_err is not None:
            self._ra_err = ra_err

        if dec_err is not None:
            self._dec_err = dec_err
            
            
    def _sep_bright_sources(self):

        #for the case of certification errror
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        #read in table from website   
        url = "https://swift.gsfc.nasa.gov/results/transients/BAT_current.html" 
        table_MN = pd.read_html(url)
        df = table_MN[0]

        #delete unwanted string
        df.rename(columns={'Peak*': 'Peak'}, inplace=True)

        #filter by peak value
        df = df.drop(df[df.Peak=="-"].index)
        df = df.astype({"Peak": int})
        df_filtered = df[df['Peak']>400]
        
        #for table of catalog
        table = Table.from_pandas(df_filtered)
        #table.show_in_browser(jsviewer=True)
        
        #transform input in SkyCoord
        position = SkyCoord(self._ra*unit.deg, self._dec*unit.deg, frame="icrs")
        
        #transform table data in SkyCoord
        coords = []
        for i in range(len(df_filtered['RA J2000 Degs'])):
            ra = table[i]['RA J2000 Degs']  
            dec = table[i]['Dec J2000 Degs']
            coords.append(SkyCoord(ra*unit.deg, dec*unit.deg, frame="icrs"))
            
        #get separation value
        separations = []
        for i in coords:
            z = i.separation(position)
            separations.append(z.to(unit.deg))
            
        #for table of separations
        table["Separation Degs"] = separations
        table.round(3)
        table.sort("Separation Degs")
        #table.show_in_browser(jsviewer=True)
            
        #create dictionary
        dic = {}
        for i in range(3):
            dic[table[i]['Source Name']]={"ra":table[i]['RA J2000 Degs'],
            "dec":table[i]['Dec J2000 Degs'], "separation":table[i]["Separation Degs"]}

        self._dic_bright_sources = dic


    def _sep_SGRs(self):

        #get csv data from website
        url = 'http://www.physics.mcgill.ca/~pulsar/magnetar/TabO1.csv'
        r = requests.get(url, allow_redirects=True)
        open('SGRList.csv', 'wb').write(r.content)
        df = pd.read_csv('SGRList.csv')

        #for table of catalog
        table = Table.from_pandas(df)
    
        #transform table data in SkyCoord
        coords = []
        for i in range(len(df['RA'])):
            hour_ra = Angle(table[i]['RA']+" hours")
            ra = hour_ra.to(unit.deg)
            arc_dec = table[i]['Decl']
            dec = Angle(tuple(map(float,arc_dec.split(' '))), unit=unit.deg)
            coords.append(SkyCoord(ra, dec, frame="icrs"))
                
        #transform input in SkyCoord
        position=SkyCoord(self._ra*unit.deg, self._dec*unit.deg, frame="icrs")
        
        #get separation value
        separations = []
        for i in coords:
            z = i.separation(position)
            separations.append(z.to(unit.deg))
        
        #for table of separations
        table["Separation Degs"] = separations
        table["Coords Degs"] = coords
        table.round(3)
        table.sort("Separation Degs")
        #table.show_in_browser(jsviewer=True)
        
        #create dictionary
        dic = {}
        for i in range(3):
            dic[table[i]['Name']]={"ra":round(table[i]['Coords Degs'].ra.degree,3),
            "dec":round(table[i]['Coords Degs'].dec.degree,3),
            "separation":table[i]["Separation Degs"]}

        self._dic_SGRs = dic
    

    def _build_report(self):
        self._report = {
            "general": {
                "grb_name": f"{self.grb_name}",
                "grb_name_gcn": f"{self._grb_name_gcn}",
                "report_type": f"{self.report_type}",
                "version": f"{self.version}",
                "trigger_number": self._trigger_number,
                "trigger_timestamp": self._trigger_timestamp,
                "data_timestamp": self._data_timestamp,
                "localization_timestamp": datetime.utcnow().strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "most_likely": self._most_likely,
                "second_most_likely": self._second_most_likely,
                "swift": self._swift,
            },
            "fit_result": {
                "model": self._model,
                "ra": convert_to_float(self._ra),
                "ra_err": convert_to_float(self._ra_err),
                "dec": convert_to_float(self._dec),
                "dec_err": convert_to_float(self._dec_err),
                "spec_K": convert_to_float(self._K),
                "spec_K_err": convert_to_float(self._K_err),
                "spec_index": convert_to_float(self._index),
                "spec_index_err": convert_to_float(self._index_err),
                "spec_xc": convert_to_float(self._xc),
                "spec_xc_err": convert_to_float(self._xc_err),
                "spec_alpha": convert_to_float(self._alpha),
                "spec_alpha_err": convert_to_float(self._alpha_err),
                "spec_xp": convert_to_float(self._xp),
                "spec_xp_err": convert_to_float(self._xp_err),
                "spec_beta": convert_to_float(self._beta),
                "spec_beta_err": convert_to_float(self._beta_err),
                "sat_phi": convert_to_float(self._phi_sat),
                "sat_theta": convert_to_float(self._theta_sat),
                "balrog_one_sig_err_circle": convert_to_float(
                    self._balrog_one_sig_err_circle
                ),
                "balrog_two_sig_err_circle": convert_to_float(
                    self._balrog_two_sig_err_circle
                ),
            },
            "time_selection": {
                "bkg_neg_start": self._bkg_neg_start,
                "bkg_neg_stop": self._bkg_neg_stop,
                "bkg_pos_start": self._bkg_pos_start,
                "bkg_pos_stop": self._bkg_pos_stop,
                "active_time_start": self._active_time_start,
                "active_time_stop": self._active_time_stop,
                "used_detectors": self._used_detectors,
            },
            "separation_values": {
                "bright_sources": self._dic_bright_sources,
                "SGRs": self._dic_SGRs
            }
        }


    def save_result_yml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self._report, f, default_flow_style=False)

    def __repr__(self):
        """
        Examine the balrog results.
        """

        print(f"Result Reader for {self.grb_name}")
        return self._report

    @property
    def ra(self):

        return self._ra, self._ra_err

    @property
    def dec(self):

        return self._dec, self._dec_err

    @property
    def K(self):

        return self._K, self._K_err

    @property
    def alpha(self):

        return self._alpha, self._alpha_err

    @property
    def xp(self):

        return self._xp, self._xp_err

    @property
    def beta(self):

        return self._beta, self._beta_err

    @property
    def index(self):

        return self._index, self._index_err

    @property
    def xc(self):

        return self._xc, self._xc_err

    @property
    def model(self):

        return self._model


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


def check_letter(trigger_number, grb_name):
    """
    Method to get the GRB Letter from the GCN Archive
    """
    try:
        day = grb_name[3 : 3 + 6]
        gcn_ids = []
        used_names = []
        final_name = "???"

        # Get the GCN archive
        page = urllib.request.urlopen("https://gcn.gsfc.nasa.gov/gcn3_archive.html")

        # Iterate over all gcn titles
        for line in page.readlines():
            line_str = line.decode()

            # Check if the day and "GRB" appear in the line
            if day in line_str and "GRB " in line_str:
                # Get the id of the GCN notice and the gcn_name
                gcn_id = line_str.split("</A>")[0][-5:]
                gcn_grb_name = line_str.split("GRB ")[1][:7]

                # Check if its a Fermi GBM GCN
                if "Fermi GBM" in line_str:

                    # Open the GCN and check if its the same trigger_number
                    gcn = urllib.request.urlopen(
                        "https://gcn.gsfc.nasa.gov/gcn3/{}.gcn3".format(gcn_id)
                    )

                    same_trigger = False

                    for gcn_line in gcn.readlines():
                        gcn_line_str = gcn_line.decode()

                        if str(trigger_number) in gcn_line_str:
                            # We found a Fermi GBM GCN with the same trigger number
                            # So we can use their gcn_grb_name
                            final_name = gcn_grb_name
                            break

                gcn_ids.append(gcn_id)
                used_names.append(gcn_grb_name)

    except Exception as e:
        print(e)
        final_name = "???"

    if final_name == "???":

        # Check if there have been GRBs during that day
        if len(used_names) > 0:
            # Get unique values by converting to set and back to list
            # Then sort this list
            used_names = list(set(used_names))
            used_names = sorted(used_names)

            # Now we need to correlate the times to check if its the same GRB
            # This is currently not implemented and has to be done by hand!
            print(used_names)

        return final_name
    else:
        return "GRB{}".format(final_name)


def get_best_fit_with_errors(post_equal_weigts_file, model):
    """
    load fit results and get best fit and errors
    :return:
    """
    chain = loadtxt2d(post_equal_weigts_file)

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

    # Calculate err radius #
    chains, parameters, truth, extents, blind, log_scales = c2.plotter._sanitise(
        None, None, None, None, color_p=True, blind=None
    )

    summ = c2.analysis.get_summary(
        parameters=["ra (deg)", "dec (deg)"], chains=chains, squeeze=False
    )[0]
    ra = summ["ra (deg)"][1]
    try:
        ra_pos_err = summ["ra (deg)"][2] - summ["ra (deg)"][1]
        ra_neg_err = summ["ra (deg)"][1] - summ["ra (deg)"][0]

        if np.absolute(ra_pos_err) > np.absolute(ra_neg_err):
            ra_err = np.absolute(ra_pos_err)
        else:
            ra_err = np.absolute(ra_neg_err)

    except:
        ra_err = None

    dec = summ["dec (deg)"][1]

    try:
        dec_pos_err = summ["dec (deg)"][2] - summ["dec (deg)"][1]
        dec_neg_err = summ["dec (deg)"][1] - summ["dec (deg)"][0]

        if np.absolute(dec_pos_err) > np.absolute(dec_neg_err):
            dec_err = np.absolute(dec_pos_err)
        else:
            dec_err = np.absolute(dec_neg_err)

    except:
        dec_err = None

    hist, x_contour, y_contour = c2.plotter._get_smoothed_histogram2d(
        chains[0], "ra (deg)", "dec (deg)"
    )  # ra, dec in deg here

    hist[hist == 0] = 1e-16
    val_contour = c2.plotter._convert_to_stdev(hist.T)

    mask = val_contour < 0.68
    points = []
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                points.append([x_contour[j], y_contour[i]])
    points = np.array(points)
    best_fit_point = [ra, dec]
    best_fit_point_vec = [
        np.cos(best_fit_point[1] * np.pi / 180)
        * np.cos(best_fit_point[0] * np.pi / 180),
        np.cos(best_fit_point[1] * np.pi / 180)
        * np.sin(best_fit_point[0] * np.pi / 180),
        np.sin(best_fit_point[1] * np.pi / 180),
    ]
    alpha_largest = 0

    for point_2 in points:
        point_2_vec = [
            np.cos(point_2[1] * np.pi / 180) * np.cos(point_2[0] * np.pi / 180),
            np.cos(point_2[1] * np.pi / 180) * np.sin(point_2[0] * np.pi / 180),
            np.sin(point_2[1] * np.pi / 180),
        ]
        alpha = np.arccos(np.dot(point_2_vec, best_fit_point_vec)) * 180 / np.pi
        if alpha > alpha_largest:
            alpha_largest = alpha
    alpha_one_sigma = alpha

    mask = val_contour < 0.95
    points = []
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                points.append([x_contour[j], y_contour[i]])
    points = np.array(points)
    alpha_largest = 0

    for point_2 in points:
        point_2_vec = [
            np.cos(point_2[1] * np.pi / 180) * np.cos(point_2[0] * np.pi / 180),
            np.cos(point_2[1] * np.pi / 180) * np.sin(point_2[0] * np.pi / 180),
            np.sin(point_2[1] * np.pi / 180),
        ]
        alpha = np.arccos(np.dot(point_2_vec, best_fit_point_vec)) * 180 / np.pi
        if alpha > alpha_largest:
            alpha_largest = alpha
    alpha_two_sigma = alpha

    return ra, ra_err, dec, dec_err, alpha_one_sigma, alpha_two_sigma


def loadtxt2d(intext):
    try:
        return np.loadtxt(intext, ndmin=2)
    except:
        return np.loadtxt(intext)


def convert_to_float(value):
    if value is not None:
        return float(value)
    else:
        return None
