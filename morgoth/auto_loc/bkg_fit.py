import os

import gbm_drm_gen as drm
import numpy as np
import yaml
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.time_series.event_list import EventListWithDeadTime
from trigdat_reader import *

from morgoth.utils import file_utils

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")

_gbm_detectors = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']


class BkgFittingTrigdat(object):

    def __init__(self, grb_name, version, trigdat_file, time_selection_file_path):
        """
        Object used for fitting of the background in every detector and echan.
        :param grb_name: Name of GRB
        :param version: Version number of data
        :param time_selection_file_path: Path to yaml file with time selection information
        """
        self._grb_name = grb_name
        self._version = version
        self._trigdat_file = trigdat_file
        self._time_selection_file_path = time_selection_file_path

        self._build_bkg_plugins()
        self._choose_dets()

    def _build_bkg_plugins(self):
        """
        Build the background plugins for all dets
        :return:
        """
        # Time selection from yaml file
        with open(self._time_selection_file_path, 'r') as f:
            data = yaml.safe_load(f)
            active_time = data['active_time']
            bkg_time_neg = data['background_time']['before']
            bkg_time_pos = data['background_time']['after']
            poly_order = data['poly_order']

        self._trig_reader = TrigReader(self._trigdat_file,
                                       fine=False,
                                       verbose=False,
                                       poly_order=poly_order)

        self._trig_reader.set_active_time_interval(f"{active_time['start']}-{active_time['stop']}")
        self._trig_reader.set_background_selections(f"{bkg_time_neg['start']}-{bkg_time_neg['stop']}",
                                                    f"{bkg_time_pos['start']}-{bkg_time_pos['stop']}")

        self._trigdat_time_series = self._trig_reader._time_series

        self._trigdat_plugins = self._trig_reader.to_plugin(*_gbm_detectors)

    def save_lightcurves(self, dir_path):
        """
        Save plots of the lightcurves for all dets 
        :param dir_path: Directory path where to save the plots
        :return:
        """

        # Max time from yaml file
        with open(self._time_selection_file_path, 'r') as f:
            data = yaml.load(f)
            max_time = data['max_time']

        file_utils.if_dir_containing_file_not_existing_then_make(dir_path)

        plots = self._trig_reader.view_lightcurve(start=-150, stop=float(max_time), return_plots=True)

        self._lightcurve_plots = {}

        for det_name, fig in plots:
            file_path = os.path.join(dir_path, f"{self._grb_name}_lightcurve_trigdat_detector_{det_name}_plot_{self._version}.png")

            fig.savefig(file_path, bbox_inches='tight')
            self._lightcurve_plots[det_name] = file_path

    def save_bkg_file(self, dir_path):
        """
        Save the hdf5 file with background polynom information        
        :param dir_path: Directory path where to save the bkg h5 file
        :return:
        """
        file_utils.if_dir_containing_file_not_existing_then_make(dir_path)

        self._bkg_fits_files = {}

        for det_name in _gbm_detectors:
            file_path = os.path.join(dir_path, f'bkg_det_{det_name}.h5')

            self._trigdat_time_series[det_name].save_background(file_path)
            self._bkg_fits_files[det_name] = file_path

    def _choose_dets(self):
        """
        Function to automatically choose the detectors which should be used in the fit
        :return:
        """
        # create plugin with the detectors you wish to use for the fit
        sign_list = []
        ndet = 7

        # get significance of all the detectors
        for i, det in enumerate(self._trigdat_plugins):
            sign_list.append([det.significance, i])
        sign_list = np.array(sign_list)

        # delete nan entries
        sign_list = sign_list[~np.isnan(sign_list[:, 0])]

        # get index with the most significance
        sign_list_sort = sign_list[sign_list[:, 0].argsort()]

        index_sign_max = sign_list_sort[-1, 1]

        side_1_indices = [0, 1, 2, 3, 4, 5, 12]
        side_2_indices = [6, 7, 8, 9, 10, 11, 13]

        # only use the detectors on the same side as the detector with the most significance
        if index_sign_max in side_1_indices:

            self._use_dets = side_1_indices

        else:
            self._use_dets = side_2_indices

        print('Detectors for fit: ', self._use_dets)

    def save_yaml(self, path):
        """
        Save yaml with needed information.
        :param path: Path where to save the yaml
        :return:
        """
        file_utils.if_dir_containing_file_not_existing_then_make(path)

        bkg_fit_dict = {}
        bkg_fit_dict['use_dets'] = self._use_dets
        bkg_fit_dict['bkg_fit_files'] = self._bkg_fits_files
        bkg_fit_dict['lightcurve_plots'] = self._lightcurve_plots

        with open(path, "w") as outfile:
            yaml.dump(bkg_fit_dict, outfile, default_flow_style=False)

    @property
    def use_dets(self):

        return self._use_dets

    def set_used_dets(self, det_list):
        """
        Supports both det countings. 10-13 and n0-b1.
        """
        det_list_final = []
        for det in det_list:
            print(det)
            if type(det) == int:

                det_list_final.append(det)

            elif 'n' in det:

                if det[1] == 'a':

                    det_list_final.append(10)

                elif det[1] == 'b':

                    det_list_final.append(11)

                else:

                    det_list_final.append(int(det[1]))

            elif 'b0' in det:

                det_list_final.append(12)

            elif 'b1' in det:

                det_list_final.append(13)

            else:

                raise Exception('Wrong format for detector selection')

        self._use_dets = det_list_final


class BkgFittingTTE(object):

    def __init__(self, grb_name, version, time_selection_file_path, bkg_fitting_file_path):
        """
        Object used for fitting of the background in every detector and echan for tte data.
        :param grb_name: Name of GRB
        :param version: Version number of data
        :param time_selection_file_path: Path to yaml file with time selection information
        :param bkg_fitting_file_path: Path to yaml file with information for trigdat bkg fitting
        """
        self._grb_name = grb_name
        self._version = version
        self._time_selection_file_path = time_selection_file_path
        self._trigdat_bkg_fitting_path = bkg_fitting_file_path

        self._build_bkg_plugins()
        # self._choose_dets()

    def _build_bkg_plugins(self):
        """
        Build the background plugins for all dets
        :return:
        """
        # Time selection from yaml file
        with open(self._time_selection_file_path, 'r') as f:
            data = yaml.safe_load(f)

            active_time = f"{data['active_time']['start']}-{data['active_time']['stop']}"
            background_time_neg = f"{data['background_time']['before']['start']}-{data['background_time']['before']['stop']}"
            background_time_pos = f"{data['background_time']['after']['start']}-{data['background_time']['after']['stop']}"
            poly_order = data['poly_order']

        det_ts = []

        for i in range(3):
            trigdat_file = os.path.join(base_dir, self._grb_name, f"glg_trigdat_all_bn{self._grb_name[3:]}_v0{i}.fit")
            if os.path.exists(trigdat_file):
                break

        for det in _gbm_detectors:
            tte_file = f"{base_dir}/{self._grb_name}/glg_tte_{det}_bn{self._grb_name[3:]}_{self._version}.fit"
            cspec_file = f"{base_dir}/{self._grb_name}/glg_cspec_{det}_bn{self._grb_name[3:]}_{self._version}.pha"

            # Response Setup

            rsp = BALROG_DRM(drm.DRMGenTTE(tte_file=tte_file, trigdat=trigdat_file, mat_type=2, cspecfile=cspec_file), 0.0, 0.0)

            # Time Series
            gbm_tte_file = GBMTTEFile(tte_file)
            event_list = EventListWithDeadTime(arrival_times=gbm_tte_file.arrival_times - \
                                                             gbm_tte_file.trigger_time,
                                               measurement=gbm_tte_file.energies,
                                               n_channels=gbm_tte_file.n_channels,
                                               start_time=gbm_tte_file.tstart - \
                                                          gbm_tte_file.trigger_time,
                                               stop_time=gbm_tte_file.tstop - \
                                                         gbm_tte_file.trigger_time,
                                               dead_time=gbm_tte_file.deadtime,
                                               first_channel=0,
                                               instrument=gbm_tte_file.det_name,
                                               mission=gbm_tte_file.mission,
                                               verbose=True)

            ts = TimeSeriesBuilder(det,
                                   event_list,
                                   response=rsp,
                                   poly_order=poly_order,
                                   unbinned=False,
                                   verbose=True,
                                   container_type=BinnedSpectrumWithDispersion)

            ts.set_background_interval(background_time_neg, background_time_pos)
            ts.set_active_time_interval(active_time)
            det_ts.append(ts)

        # Get start and stop of active

        # split_activetime = active_time.split('-')
        # if split_activetime[0]=='':
        #    start = -float(split_activetime[1])
        #    if split_activetime[2]=='':
        #        stop = -float(split_activetime[3])
        #    else:
        #        stop = float(split_activetime[2])
        # else:
        #    start = float(split_activetime[0])
        #    stop = float(split_activetime[1])

        # det_ts[-1].create_time_bins(start=start,stop=stop, method='constant', dt=0.1)
        # tstart = ts.bins.starts
        # tstop = ts.bins.stops
        # active_time = '{}-{}'.format(tstart[0],tstop[0])
        # det_ts[-1].set_active_time_interval(active_time)
        # for ts in det_ts[:-1]:
        #    ts.set_active_time_interval(active_time)
        #    ts.create_time_bins(start=tstart, stop=tstop, method='custom')

        self._ts = det_ts

    def save_lightcurves(self, dir_path):
        """
        Save plots of the lightcurves for all dets 
        :param dir_path: Directory path where to save the plots
        :return:
        """
        # Max time from yaml file
        with open(self._time_selection_file_path, 'r') as f:
            data = yaml.load(f)
            max_time = data['Max_Time']

        file_utils.if_dir_containing_file_not_existing_then_make(dir_path)

        self._lightcurve_plots = {}

        for det_name, ts in zip(_gbm_detectors, self._ts):
            file_path = os.path.join(dir_path, f"{self._grb_name}_lightcurve_tte_detector_{det_name}_plot_{self._version}.png")

            fig = ts.view_lightcurve(start=-150, stop=float(max_time))
            fig.savefig(file_path, dpi=350, bbox_inches='tight')

            self._lightcurve_plots[det_name] = file_path

    def save_bkg_file(self, dir_path):
        """
        Save the hdf5 file with background polynom information        
        :param dir_path: Directory path where to save the bkg h5 file
        :return:
        """

        file_utils.if_dir_containing_file_not_existing_then_make(dir_path)

        self._bkg_fits_files = {}

        for i, det_name in enumerate(_gbm_detectors):
            file_path = os.path.join(dir_path, f'bkg_det_{det_name}.h5')
            self._ts[i].save_background(file_path)
            self._bkg_fits_files[det_name] = file_path

    def save_yaml(self, path):
        """
        Save yaml with needed information.
        :param path: Path where to save the yaml
        :return:
        """
        file_utils.if_dir_containing_file_not_existing_then_make(path)

        bkg_fit_dict = {}
        bkg_fit_dict['bkg_fits_files'] = self._bkg_fits_files
        bkg_fit_dict['lightcurve_plots'] = self._lightcurve_plots
        with open(self._trigdat_bkg_fitting_path, "r") as f:
            bkg_fit_dict["use_dets"] = yaml.safe_load(f)["use_dets"]

        with open(path, "w") as f:
            yaml.dump(bkg_fit_dict, f, default_flow_style=False)

    @property
    def use_dets(self):

        return self._use_dets

    def set_used_dets(self, det_list):
        """
        Supports both det countings. 10-13 and n0-b1.
        """
        det_list_final = []
        for det in det_list:
            print(det)
            if type(det) == int:

                det_list_final.append(det)

            elif 'n' in det:

                if det[1] == 'a':

                    det_list_final.append(10)

                elif det[1] == 'b':

                    det_list_final.append(11)

                else:

                    det_list_final.append(int(det[1]))

            elif 'b0' in det:

                det_list_final.append(12)

            elif 'b1' in det:

                det_list_final.append(13)

            else:

                raise Exception('Wrong format for detector selection')

        self._use_dets = det_list_final
