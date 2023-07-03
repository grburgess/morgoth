from operator import length_hint
import os

import yaml
import numpy as np

from astropy.stats import bayesian_blocks

from morgoth.utils.trig_reader import TrigReader
from morgoth.auto_loc.utils.functions_for_auto_loc import *

from threeML.utils.statistics.stats_tools import Significance

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelection(object):
    def __init__(self, grb_name, trigdat_file, fine=False):
        """
        Object for time selection of a GRB event.
        :param grb_name: Name of GRB
        :param version: Version of trigdat data file
        """
        self._fine = fine
        self._grb_name = grb_name
        self._trigdat_file = trigdat_file

        self.trigdata_time_selection()

    def trigdata_time_selection(self):
        """
        Function to calcuate the time selection for a given trigger event. This is done iterative.
        :return:
        """
        trig_reader = TrigReader(self._trigdat_file, fine=self._fine, verbose=False)

        self.trig_reader = trig_reader
        # Inital bkg and active time selection - We will change this recursivly to explain most of
        # the bkg by a polynomial
        trig_reader.set_active_time_interval("0-0")
        trig_reader.set_background_selections("-150-0", "0-150")

        # In several steps cut away data from background section that are at least in one
        # detector above the sigma_lim value

        sigma_lims = [100, 50, 30, 10, 7, 5, 3]
        for sigma_lim in sigma_lims:
            (
                bkg_neg_start,
                bkg_neg_stop,
                bkg_pos_start,
                bkg_pos_stop,
                active_time_start,
                active_time_stop,
                max_time,
            ) = get_new_intervals(sigma_lim, trig_reader)

            # set active_time new and set background selection new => new background fit with this selection
            trig_reader.set_active_time_interval(
                f"{active_time_start}-{active_time_stop}"
            )
            trig_reader.set_background_selections(
                f"{bkg_neg_start}-{bkg_neg_stop}", f"{bkg_pos_start}-{bkg_pos_stop}"
            )

        self._bkg_neg_start = bkg_neg_start
        self._bkg_neg_stop = bkg_neg_stop
        self._bkg_pos_start = bkg_pos_start
        self._bkg_pos_stop = bkg_pos_stop
        self._active_time_start = active_time_start
        self._active_time_stop = active_time_stop

        self._background_time_neg = f"{bkg_neg_start}-{bkg_neg_stop}"
        self._background_time_pos = f"{bkg_pos_start}-{bkg_pos_stop}"
        self._active_time = f"{active_time_start}-{active_time_stop}"
        self._max_time = max_time

        self._poly_order = -1

    def save_yaml(self, path):
        """
        Save the automatic time selection in a yaml file
        :param path: Path where to save the yaml file
        :return:
        """
        time_select = {
            "active_time": {
                "start": self._active_time_start,
                "stop": self._active_time_stop,
            },
            "background_time": {
                "before": {"start": self._bkg_neg_start, "stop": self._bkg_neg_stop},
                "after": {"start": self._bkg_pos_start, "stop": self._bkg_pos_stop},
            },
            "max_time": self._max_time,
            "poly_order": self._poly_order,
            "fine": self._fine,
        }

        # Poly_Order entry with -1 (default). But we need this entry in the
        # yaml file to give us the possibility to alter it if we see that it screws up one of
        # the det fits

        with open(path, "w") as outfile:
            yaml.dump(time_select, outfile, default_flow_style=False)

    @property
    def background_time_neg(self):
        return self._background_time_neg

    @property
    def background_time_pos(self):
        return self._background_time_pos

    @property
    def active_time(self):
        return self._active_time

    @property
    def max_time(self):
        return self._max_time

    def set_background_time_pos(self, tstart=None, tstop=None, string=None):
        assert string is None or (
            tstart is None and tstop is None
        ), "Only use string definition or start and stop time definition!"
        assert string is not None or (
            tstart is not None and tstop is not None
        ), "String definition and start and stop time are both set to None!"

        if string is not None:
            self._background_time_pos = string
        else:
            self._background_time_pos = "{}-{}".format(tstart, tstop)

    def set_background_time_neg(self, tstart=None, tstop=None, string=None):
        assert string is None or (
            tstart is None and tstop is None
        ), "Only use string definition or start and stop time definition!"
        assert string is not None or (
            tstart is not None and tstop is not None
        ), "String definition and start and stop time are both set to None!"

        if string is not None:
            self._background_time_neg = string
        else:
            self._background_time_neg = f"{tstart}-{tstop}"

    def set_active_time(self, tstart=None, tstop=None, string=None):
        assert string is None or (
            tstart is None and tstop is None
        ), "Only use string definition or start and stop time definition!"
        assert string is not None or (
            tstart is not None and tstop is not None
        ), "String definition and start and stop time are both set to None!"

        if string is not None:
            self._active_time = string
        else:
            self._active_time = f"{tstart}-{tstop}"

    def set_max_time(self, max_time):
        self._max_time = max_time


class TimeSelectionKnown(TimeSelection):
    def __init__(
        self,
        active_time,
        background_time_neg,
        background_time_pos,
        poly_order=-1,
        max_time=None,
        fine=False,
    ):
        self._fine = fine
        self._active_time = active_time
        self._background_time_neg = background_time_neg
        self._background_time_pos = background_time_pos
        self._max_time = max_time
        self._poly_order = poly_order

        # get start and stop times
        inter = self._active_time.split("-")
        if len(inter) == 2:
            self._active_time_start = float(inter[0])
            self._active_time_stop = float(inter[1])
        elif len(inter) == 3:
            self._active_time_start = -1 * float(inter[1])
            self._active_time_stop = float(inter[2])
        else:
            self._active_time_start = -1 * float(inter[1])
            self._active_time_stop = -1 * float(inter[3])

        inter = self._background_time_pos.split("-")
        if len(inter) == 2:
            self._bkg_pos_start = float(inter[0])
            self._bkg_pos_stop = float(inter[1])
        elif len(inter) == 3:
            self._bkg_pos_start = -1 * float(inter[1])
            self._bkg_pos_stop = float(inter[2])
        else:
            self._bkg_pos_start = -1 * float(inter[1])
            self._bkg_pos_stop = -1 * float(inter[3])

        inter = self._background_time_neg.split("-")
        if len(inter) == 2:
            self._bkg_neg_start = float(inter[0])
            self._bkg_neg_stop = float(inter[1])
        elif len(inter) == 3:
            self._bkg_neg_start = -1 * float(inter[1])
            self._bkg_neg_stop = float(inter[2])
        else:
            self._bkg_neg_start = -1 * float(inter[1])
            self._bkg_neg_stop = -1 * float(inter[3])


class TimeSelectionBB(TimeSelection):
    """Automatically sets active trigger time as well as neg and pos (before and after trigger) background"""

    def __init__(
        self,
        grb_name,
        trigdat_file,
        fine=False,
        gamma=0.776,
        mean_factor=1.1,
        significance_dets=3,
    ):
        """Starts Timeselection

        Args:
            trigdat_file (path like): path to trigdat file
            trigger_name (str): name of the trigger
            fine (bool, optional): Use fine data binning. Defaults to False.
            gamma (float, optional): gamma value for bayesian blocks (influences number of blocks). Defaults to 0.776.
            mean_factor (float, optional): factor scaling the mean cps rate used for ruling out too long selections
        """
        self._fine = fine
        self._trigdat_file = trigdat_file

        # Load Trigdat file
        self._trigreader_obj = TrigReader(self._trigdat_file, fine=self._fine)

        # Names of detectors
        self.dets = [
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

        # GRB name
        self.trigger_name = grb_name

        # Gamma Value for BB - default to 0.776
        self._gamma = gamma

        # Mean Value Factor for elminiating too long selections - defaults to 1.0
        self._mean_factor = mean_factor

        self._significance_dets = significance_dets
        # initialize dicts
        self._bayesian_block_edges_dict = {}
        self._bayesian_block_times_dict = {}
        self._bayesian_block_cps_dict = {}
        self._bayesian_block_widths_dict = {}

        # contains start/stop times of trigger for each det
        self._start_trigger_dict = {}
        self._stop_trigger_dict = {}

        self.pos_bkg_dict = {}  # start/stop of bkg after trigger for each det
        self.neg_bkg_dict = {}  # start/stop of bkg before trigger for each det
        self._bkg_list_dict = {}  # lists containing pos and neg bkg for each det
        self._significance_dict = {}

        # used for selecting detectors for finalizing times

        self._poly_order = -1

        # get relevant data from Trigdat object
        self._processTrigdat()

        # run timeselection on every detector
        self.timeselection()

        # combine most significant lightcurves
        self.fixSelections()

    @property
    def trigreader_object(self):
        """TrigReader Object

        Returns:
            TrigReader: TrigReader Object
        """
        return self._trigreader_obj

    @property
    def start_trigger_dict(self):
        """Start Trigger dictionary for every detector

        Returns:
            dict: Start Trigger for each det
        """
        return self._start_trigger_dict

    @property
    def stop_trigger_dict(self):
        """Stop Trigger dictionary for every detector

        Returns:
            dict: Stop Trigger for each det
        """
        return self._stop_trigger_dict

    @property
    def bayesian_block_times_dict(self):
        """Dict containing the start times of the blocks

        Returns:
            dict: bb start times
        """
        return self._bayesian_block_times_dict

    @property
    def bayesian_block_widths_dict(self):
        """Dict containg the widths of the blocks

        Returns:
            dict: bb block widths
        """
        return self._bayesian_block_widths_dict

    @property
    def bayesian_block_cps_dict(self):
        """Dict containing the cps of the blocks

        Returns:
            dict: bb cps
        """
        return self._bayesian_block_cps_dict

    @property
    def start_trigger(self):
        """Final start time of the active phase

        Returns:
            float: start of active phase
        """
        return self._active_time_start

    @property
    def stop_trigger(self):
        """Final stop time of active phase

        Returns:
            floast: stop of active phase
        """
        return self._active_time_stop

    @property
    def detector_selection(self):
        """Detectors with highest significance
        Returns:
             str: dets
        """
        return self._detector_selection

    def timeselection(
        self,
        lower_trigger_bound=-10,
        upper_trigger_bound=50,
        max_trigger_length=10.5,
    ):
        """runs timeselection for each detector in self.dets individually

        Args:
            lower_trigger_bound (int, optional): lower bound for trigger time. Defaults to -4.
            upper_trigger_bound (int, optional): upper bound for trigger time. Defaults to 50.
            max_trigger_length (float, optional): max length of trigger time. Keep in mind that for times >10s the detector moves about ~1 deg. Defaults to 10.24.
        """
        self._lower_trigger_bound = lower_trigger_bound
        self._upper_trigger_bound = upper_trigger_bound
        self._max_trigger_length = max_trigger_length

        for det in self.dets:
            print(f"Run Timeselection for detector {det}")
            # calculate BB for det
            self._bayesianBlocks(det)

            # get the best start-stop of active time for the det
            self._calcStartStopTrigger(det)
            self.neg_bkg_dict[det] = []
            self.pos_bkg_dict[det] = []
            self._bkg_list_dict[det] = []

            # calculate the background for det
            bkgSelectorDets = BackgroundSelector(self, det)
            bkgSelectorDets.runSelector()

            # calculate the significance during the trigger time
            self._getSignificance(det)

    def fixSelections(self):
        """Improves the timeselection by combining multiple detectors with the highest significance in the data

        Args:
            significance_dets_nr (int, optional): Number of detectors which will be used for combined timeselection. Defaults to 3.
        """
        significance_dets_nr = self._significance_dets
        significance_max = {}

        for det, sign in self._significance_dict.items():
            # set the significance outside the trigger selection to zero to prevent "false" high significance
            sign[: self._times.index(self._start_trigger_dict[det])] = 0
            sign[self._times.index(self._stop_trigger_dict[det]) :] = 0
            max_det = np.max(sign)
            significance_max[det] = max_det

        # sort the significance_max dict in descending order
        significance_max = dict(
            sorted(significance_max.items(), key=lambda item: item[1], reverse=True)
        )

        # empty list and string for combination of detectors
        obs_combined = []
        det = ""
        self._detector_selection_list = []
        for i in range(significance_dets_nr):
            det += list(significance_max.keys())[i]
            self._detector_selection_list.append(list(significance_max.keys())[i])
            obs_combined.append(self._cps_dets[list(significance_max.keys())[i]])
        self._detector_selection = det
        print(
            f"The detecors {det} had the highest significance and are used to fix the selections"
        )
        obs_combined = np.sum(obs_combined, axis=0)
        self._cps_dets[det] = obs_combined
        self._bayesianBlocks(det)

        # rebinning and timeselection
        (
            self._bayesian_block_times_dict[det],
            self._bayesian_block_cps_dict[det],
            self._bayesian_block_widths_dict[det],
        ) = bb_binner(self._times, obs_combined, self._bayesian_block_edges_dict[det])

        self._calcStartStopTrigger(det)
        start_trigger = self._start_trigger_dict[det]
        end_trigger = self._stop_trigger_dict[det]

        bkgSelector = BackgroundSelector(self, det)
        bkgSelector.runSelector()
        before_trigger = self.neg_bkg_dict[det]
        after_trigger = self.pos_bkg_dict[det]
        background_sel = [before_trigger, after_trigger]

        self._background_time_pos = f"{background_sel[1][0]}-{background_sel[1][1]}"
        self._background_time_neg = f"{background_sel[0][0]}-{background_sel[0][1]}"

        self._active_time = f"{start_trigger}-{end_trigger}"

        self._active_time_start = start_trigger
        self._active_time_stop = end_trigger

        self._bkg_neg_start = float(background_sel[0][0])
        self._bkg_neg_stop = float(background_sel[0][1])

        self._bkg_pos_start = float(background_sel[1][0])
        self._bkg_pos_stop = float(background_sel[1][1])

        self._max_time = float(background_sel[1][1])

        background_sel_strings = self._background_time_neg, self._background_time_pos

        bkgSelector._polyFit(
            background_sel_strings=background_sel_strings,
            active_sel_string=self._active_time,
        )

    def set_max_time(self, max_time):
        self._max_time = max_time
        self.timeselection()
        self.fixSelections()

    def _processTrigdat(self):
        """Loads trigdat data and stores times, observed cps and bin widths"""
        # get cps and times from Trigdat object
        obs_array, _ = self._trigreader_obj.observed_and_background()
        start_times, end_times = self._trigreader_obj.tstart_tstop()
        times_dets = start_times.tolist()
        # Fix for bayesian Blocks (add an additional block with count rate zero )
        times_dets.append(end_times[-1])
        obs_dets = []
        for i in range(len(obs_array)):
            temp = list(map(int, obs_array[i].tolist()))
            temp.append(0)
            obs_dets.append(temp)

        width = [times_dets[i + 1] - times_dets[i] for i in range(len(times_dets) - 1)]

        # set last bin length the same as the one before (approx 8s)
        width.append(width[-1])
        times_no_duplicates = []
        times_duplicates = []
        for i, t in enumerate(times_dets):
            if t not in times_no_duplicates:
                times_no_duplicates.append(t)
            else:
                times_duplicates.append(i)

        for i_d in times_duplicates:
            for det_nr in range((len(obs_dets))):
                obs_dets[det_nr].pop(i_d)

        self._times = times_no_duplicates
        self._max_time = times_dets[-1]

        self._cps_dets = {}
        self._timebin_widths = {}
        for det_nr, det in enumerate(self.dets):
            assert len(obs_dets[det_nr]) == len(
                self._times
            ), "Length of times and values does not match!"
            self._cps_dets[det] = obs_dets[det_nr]
            self._timebin_widths[det] = width[det_nr]

    def _bayesianBlocks(self, det):
        """Calculates edges of bayesian blocks and gets corresponding times, cps and widths of bins

        Args:
            det (str): Name of detector (or multiple ones)
        """
        self._bayesian_block_edges_dict[det] = bayesian_blocks(
            self._times, self._cps_dets[det], fitness="events", gamma=self._gamma
        )
        (
            self._bayesian_block_times_dict[det],
            self._bayesian_block_cps_dict[det],
            self._bayesian_block_widths_dict[det],
        ) = bb_binner(
            self._times, self._cps_dets[det], self._bayesian_block_edges_dict[det]
        )

    def _calcStartStopTrigger(self, det, max_block_width=10.24):
        """Calculates the start and stop time of the trigger

        Args:
            det (str): detector name
            max_bloc_width (optional, float): maximum allowed block width - otherwise not taken into account
        """

        start_trigger = 0
        end_trigger = 0

        cps_temp = self._get_cps_temp(det, max_block_width)
        id_max_cps_bb = np.argmax(cps_temp)
        if (
            self._bayesian_block_times_dict[det][id_max_cps_bb]
            + self._max_trigger_length
            > self._upper_trigger_bound
        ):
            ub = (
                self._bayesian_block_times_dict[det][id_max_cps_bb]
                + self._max_trigger_length
            )
            cps_temp = self._get_cps_temp(det, max_block_width, upper_bound=ub)
            id_max_cps_bb = np.argmax(cps_temp)
        if (
            self._bayesian_block_times_dict[det][id_max_cps_bb]
            - self._max_trigger_length
            < self._lower_trigger_bound
        ):
            lb = (
                self._bayesian_block_times_dict[det][id_max_cps_bb]
                - self._max_trigger_length
            )
            cps_temp = self._get_cps_temp(det, max_block_width, lower_bound=lb)
            id_max_cps_bb = np.argmax(cps_temp)

        # start length
        length_in = float(self._bayesian_block_widths_dict[det][id_max_cps_bb])
        id_l = int(id_max_cps_bb)  # index of the satrting bin
        id_h = int(id_max_cps_bb)  # index of the stopping bin

        # print(f"Lets mess things up for {det}")
        # iteratively get new length
        while True:
            length_out, id_l, id_h = self._getNewLength(length_in, id_l, id_h, det)
            if length_in == length_out:
                break
            else:
                length_in = length_out

        start_trigger = self._bayesian_block_times_dict[det][id_l]
        end_trigger = self._bayesian_block_times_dict[det][id_h + 1]

        # if the start time is still larger than end time (not sure if still needed) - swap them
        if start_trigger > end_trigger:
            start_trigger = end_trigger
            end_trigger = self._bayesian_block_times_dict[det][id_l]

        # get the corresponding indices of start/stop active time
        start_trigger_id, end_trigger_id = self.startStopToObsTimes(
            start_trigger=start_trigger, end_trigger=end_trigger
        )
        if start_trigger_id == end_trigger_id:
            end_trigger_id += 1

        start_trigger = self._times[start_trigger_id]
        end_trigger = self._times[end_trigger_id]

        self._start_trigger_dict[det] = start_trigger
        self._stop_trigger_dict[det] = end_trigger
        print(f"Set trigger time for det {det} to {start_trigger}-{end_trigger}")

    def _getNewLength(self, length_in, id_l, id_h, det):
        """Tries to get new length for trigger selection for bayesian blocks

        Args:
            length_in (float): current length of trigger
            id_l (int): current start id of trigger
            id_h (int): current end id of trigger
            det (str): detector name

        Returns:
            (float, int, int): new trigger length, start id, end id
        """
        min_id, max_id = self.startStopToObsTimes(
            self._lower_trigger_bound, self._upper_trigger_bound
        )
        min_id_start, max_id_stop = self.startStopToObsTimes(
            self._lower_trigger_bound - 20, self._upper_trigger_bound + 20
        )

        # create mask selecting 50s before and 50s after allowed trigger times
        mask = np.zeros_like(self._cps_dets[det])
        mask[min_id_start:min_id] = 1
        mask[max_id + 1 : max_id_stop] = 1

        if np.sum(mask) != 0:
            # caclulate the weighted average of the selected area
            try:
                mean_cps_trigger_area = np.average(
                    self._cps_dets[det] * mask,
                    weights=self._timebin_widths[det[:2]] * mask,
                )
            except ZeroDivisionError:
                mean_cps_trigger_area = np.mean(self._cps_dets[det] * mask)
        else:
            mean_cps_trigger_area = np.mean(self._cps_dets[det])

        cps_cond = mean_cps_trigger_area * self._mean_factor

        length_h = length_in + self._bayesian_block_widths_dict[det][id_h + 1]
        length_l = length_in + self._bayesian_block_widths_dict[det][id_l - 1]
        cps_h = self._bayesian_block_cps_dict[det][id_h + 1]
        cps_l = self._bayesian_block_cps_dict[det][id_l - 1]
        counts_l = (
            self._bayesian_block_cps_dict[det][id_l - 1]
            * self.bayesian_block_widths_dict[det][id_l - 1]
        )
        counts_h = (
            self._bayesian_block_cps_dict[det][id_h + 1]
            * self._bayesian_block_widths_dict[det][id_h + 1]
        )

        length_r = length_in
        id_l_r = id_l
        id_h_r = id_h

        # check if the next blocks fullfill the cps condition again

        if cps_l < cps_cond and cps_h < cps_cond:
            if (
                self._bayesian_block_cps_dict[det][id_l - 2] < cps_cond
                and self._bayesian_block_cps_dict[det][id_h + 2] < cps_cond
            ):
                length_r = length_in
                id_l_r = id_l
                id_h_r = id_h

            else:
                length_r, id_l_r, id_h_r = self._check_cps_cond(
                    length_h,
                    length_l,
                    counts_l,
                    counts_h,
                    length_in,
                    id_l,
                    id_h,
                    det,
                    cps_cond,
                )
        elif cps_l >= cps_cond and cps_h < cps_cond:
            if (
                self._bayesian_block_times_dict[det][id_l - 1]
                >= self._lower_trigger_bound
            ):
                if length_l <= self._max_trigger_length:
                    length_r = length_l
                    id_l_r = id_l - 1
                    id_h_r = id_h
                elif (
                    length_l - self._bayesian_block_widths_dict[det][id_h]
                    <= self._max_trigger_length
                    and counts_l
                    > self._bayesian_block_widths_dict[det][id_h]
                    * self._bayesian_block_cps_dict[det][id_h]
                ):
                    length_r = length_l - self._bayesian_block_widths_dict[det][id_h]
                    id_l_r = id_l - 1
                    id_h_r = id_h - 1
        elif cps_h >= cps_cond and cps_l < cps_cond:
            if (
                self._bayesian_block_times_dict[det][id_h + 1]
                <= self._upper_trigger_bound
            ):
                if length_h <= self._max_trigger_length:
                    length_r = length_h
                    id_l_r = id_l
                    id_h_r = id_h + 1
                elif (
                    length_h - self._bayesian_block_widths_dict[det][id_l]
                    <= self._max_trigger_length
                    and counts_h
                    > self._bayesian_block_widths_dict[det][id_l]
                    * self._bayesian_block_cps_dict[det][id_l]
                ):
                    length_r = length_h - self._bayesian_block_widths_dict[det][id_l]
                    id_l_r = id_l + 1
                    id_h_r = id_h + 1
        else:
            # both directions possible form cps condition
            length_r, id_l_r, id_h_r = self._check_counts(
                length_h, length_l, counts_l, counts_h, length_in, id_l, id_h, det
            )
        return length_r, id_l_r, id_h_r

    def _check_counts(
        self, length_h, length_l, counts_l, counts_h, length_in, id_l, id_h, det
    ):
        # default return stopping the selection
        length_r = length_in
        id_l_r = id_l
        id_h_r = id_h

        # if both next blocks are fully in the limit
        if (
            self._bayesian_block_times_dict[det][id_l - 1] >= self._lower_trigger_bound
            and self._bayesian_block_times_dict[det][id_h + 1]
            <= self._upper_trigger_bound
        ):
            # if both dont exceed the max length
            if (
                length_l <= self._max_trigger_length
                and length_h <= self._max_trigger_length
            ):
                if counts_l >= counts_h:
                    length_r = length_l
                    id_l_r = id_l - 1
                    id_h_r = id_h
                else:
                    length_r = length_h
                    id_l_r = id_l
                    id_h_r = id_h + 1
            # if the higher one exceeds the max length check if shift to higher would improve
            elif (
                length_l <= self._max_trigger_length
                and length_h > self._max_trigger_length
            ):
                if (
                    length_h - self._bayesian_block_widths_dict[det][id_l]
                    <= self._max_trigger_length
                ):
                    if (
                        counts_l
                        >= self._bayesian_block_cps_dict[det][id_h]
                        * self._bayesian_block_widths_dict[det][id_h]
                    ):
                        length_r = length_l
                        id_l_r = id_l - 1
                        id_h_r = id_h
                    else:
                        length_r = (
                            length_h - self._bayesian_block_widths_dict[det][id_l]
                        )
                        id_l_r = id_l + 1
                        id_h_r = id_h + 1
                else:
                    length_r = length_l
                    id_l_r = id_l - 1
                    id_h_r = id_h
            # if the lower one exceeds the max length check if shift to lower would improve
            elif (
                length_l > self._max_trigger_length
                and length_h <= self._max_trigger_length
            ):
                if (
                    length_l - self._bayesian_block_widths_dict[det][id_h]
                    <= self._max_trigger_length
                ):
                    if (
                        counts_h
                        > self._bayesian_block_cps_dict[det][id_l]
                        * self._bayesian_block_widths_dict[det][id_l]
                    ):
                        length_r = length_h
                        id_l_r = id_l
                        id_h_r = id_h + 1
                    else:
                        length_r = (
                            length_l - self._bayesian_block_widths_dict[det][id_h]
                        )
                        id_l_r = id_l - 1
                        id_h_r = id_h - 1
                else:
                    length_r = length_h
                    id_l_r = id_l_r
                    id_h_r = id_h + 1
            else:
                length_r = length_in
                id_l_r = id_l
                id_h_r = id_h
        # just lower block within allowed time
        elif (
            self._bayesian_block_times_dict[det][id_l - 1] >= self._lower_trigger_bound
            and self._bayesian_block_times_dict[det][id_h + 1]
            > self._upper_trigger_bound
        ):
            if length_l <= self._max_trigger_length:
                length_r = length_l
                id_l_r = id_l - 1
                id_h_r = id_h
        # just higher block within allowed time
        elif (
            self._bayesian_block_times_dict[det][id_l - 1] < self._lower_trigger_bound
            and self._bayesian_block_times_dict[det][id_h + 1]
            <= self._upper_trigger_bound
        ):
            if length_h <= self._max_trigger_length:
                length_r = length_h
                id_l_r = id_l
                id_h_r = id_h + 1
        # both blocks outside limit
        else:
            length_r = length_in
            id_l_r = id_l
            id_h_r = id_h

        return length_r, id_l_r, id_h_r

    def _check_cps_cond(
        self,
        length_h,
        length_l,
        counts_l,
        counts_h,
        length_in,
        id_l,
        id_h,
        det,
        cps_cond,
    ):
        # default returns stopping the selection
        length_r = length_in
        id_l_r = id_l
        id_h_r = id_h

        if (
            self._bayesian_block_cps_dict[det][id_l - 2] >= cps_cond
            and self._bayesian_block_cps_dict[det][id_h + 2] >= cps_cond
        ):
            if (
                self._bayesian_block_times_dict[det][id_h + 2]
                <= self._upper_trigger_bound
                and self._bayesian_block_times_dict[det][id_l - 2]
                >= self._lower_trigger_bound
            ):
                if (
                    length_l + self._bayesian_block_widths_dict[det][id_l - 1]
                    <= self._max_trigger_length
                    and length_h + self._bayesian_block_widths_dict[det][id_h + 1]
                    <= self._max_trigger_length
                ):
                    if counts_l >= counts_h:
                        length_r = length_l
                        id_l_r = id_l - 1
                        id_h_r = id_h
                    else:
                        length_r = length_h
                        id_l_r = id_l
                        id_h_r = id_h + 1
            elif (
                self._bayesian_block_times_dict[det][id_h + 2]
                <= self._upper_trigger_bound
                and self._bayesian_block_times_dict[det][id_l - 2]
                < self._lower_trigger_bound
            ):
                if (
                    length_h + self._bayesian_block_widths_dict[det][id_h + 1]
                    <= self._max_trigger_length
                ):
                    length_r = length_h
                    id_l_r = id_l
                    id_h_r = id_h + 1

            elif (
                self._bayesian_block_times_dict[det][id_h + 2]
                > self._upper_trigger_bound
                and self._bayesian_block_times_dict[det][id_l - 2]
                >= self._lower_trigger_bound
            ):
                if (
                    length_l + self._bayesian_block_widths_dict[det][id_l - 1]
                    <= self._max_trigger_length
                ):
                    length_r = length_l
                    id_l_r = id_l - 1
                    id_h_r = id_h
        elif (
            self._bayesian_block_cps_dict[det][id_l - 2] < cps_cond
            and self._bayesian_block_cps_dict[det][id_h + 2] >= cps_cond
            and self._bayesian_block_times_dict[det][id_h + 2]
            <= self._upper_trigger_bound
            and length_l + self._bayesian_block_widths_dict[det][id_h + 1]
            <= self._max_trigger_length
        ):
            if length_h <= self._max_trigger_length:
                length_r = length_h
                id_l_r = id_l
                id_h_r = id_h + 1
        elif (
            self._bayesian_block_widths_dict[det][id_l - 2] >= cps_cond
            and self._bayesian_block_times_dict[det][id_h + 2] < cps_cond
            and self._bayesian_block_times_dict[det][id_l - 2]
            >= self._lower_trigger_bound
            and length_l + self._bayesian_block_widths_dict[det][id_l - 1]
            <= self._max_trigger_length
        ):
            if length_l <= self._max_trigger_length:
                length_r = length_l
                id_l_r = id_l - 1
                id_h_r = id_h
        return length_r, id_l_r, id_h_r

    def viewBayesianBlockPlots(self, path):
        base = self._trigreader_obj.view_lightcurve(return_plots=True)
        for b in base:
            det = b[0]
            fig = b[1]
            ax = fig.get_axes()[0]
            ylim = ax.get_ylim()
            ax.vlines(
                self._bayesian_block_edges_dict[det][:-1],
                0,
                self._bayesian_block_cps_dict[det],
                color="magenta",
            )
            ax.set_ylim(ylim)
            fig.tight_layout()
            fig.savefig(os.path.join(path, f"bb_lightcurve_{det}.png"))

    def startStopToObsTimes(self, start_trigger, end_trigger):
        """Converts start and end trigger times to id of times list

        Args:
            start_trigger (float): trigger start time
            end_trigger (float): trigger end time

        Returns:
            (int, int): start id, end id
        """
        start_trigger_id = 0
        end_trigger_id = len(self._times) - 1
        for i, t in enumerate(self._times):
            if t < start_trigger and i > start_trigger_id:
                start_trigger_id = i
            elif t > end_trigger and i < end_trigger_id:
                end_trigger_id = i
        start_trigger_id += 1
        end_trigger_id -= 1

        return start_trigger_id, end_trigger_id

    def _getSignificance(self, det):
        """Calculates the significance for a given detector using threeML

        Args:
            det (str): detector name
        """
        self._significance_dict[det] = self._trigreader_obj.time_series[
            det
        ].significance_per_interval

    def _get_cps_temp(self, det, max_block_width, lower_bound=None, upper_bound=None):
        # copy cps to temp array for manipulation
        cps_temp = np.copy(self._bayesian_block_cps_dict[det])
        # setting first and last block to 0 (they are very small timely) as well as every block with a duration > 10s or if it's end is outside the set bounds

        cps_temp[-1] = 0
        cps_temp[0] = 0

        if lower_bound is None:
            lower_bound = self._lower_trigger_bound
        if upper_bound is None:
            upper_bound = self._upper_trigger_bound

        for i, val in enumerate(self._bayesian_block_widths_dict[det]):
            # check if count rate is already 0
            if cps_temp[i] != 0:
                # check if the width is greater than the max allowed
                if val > max_block_width:
                    cps_temp[i] = 0
                else:
                    # check if the block is inside the allowed time range
                    if (
                        self._bayesian_block_times_dict[det][i] < lower_bound
                        or self._bayesian_block_times_dict[det][i] > upper_bound
                    ):
                        cps_temp[i] = 0

        # id_max_counts_bb = np.argmax(
        #    np.array(cps_temp) * np.array(self._bayesian_block_widths_dict[det])
        # )
        return cps_temp


class BackgroundSelector:
    """Class for background selection"""

    def __init__(
        self,
        timeSelection: TimeSelectionBB,
        det,
        bkg_min_length=50,
        min_distance_trigger=20,
    ):
        """Initializes Background selection

        Args:
            timeSelection (TimeSelectionBB): TimeSelectionBB Object
            det (str): detector name
            bkg_min_length (int, optional): Minimum length of pos and neg background selection. Defaults to 50.
            min_distance_trigger (int, optional): Minimum distance to trigger. Defaults to 20.
        """
        self._bkg_bin_min_length = bkg_min_length
        self._bkg_distance_trigger = min_distance_trigger
        self._timeSelection = timeSelection
        self._det = det
        self._max_time = timeSelection.max_time

    def runSelector(self):
        """Runs the Background selection"""
        (
            self._timeSelection.neg_bkg_dict[self._det],
            self._timeSelection.pos_bkg_dict[self._det],
        ) = self._selectBackground()
        self._timeSelection._bkg_list_dict[self._det] = [
            self._timeSelection.neg_bkg_dict[self._det],
            self._timeSelection.pos_bkg_dict[self._det],
        ]
        self._polyFit()

    def _selectBackground(self):
        """Selects the pos and neg bkg times

        Returns:
            (list,list): (neg_bkg_bounds, pos_bkg_bounds)
        """

        # set the range around trigger which won#t contribute to the bkg
        start_trigger_range = (
            self._timeSelection.start_trigger_dict[self._det]
            - self._bkg_distance_trigger
        )
        end_trigger_range = (
            self._timeSelection.stop_trigger_dict[self._det]
            + self._bkg_distance_trigger
        )

        # calc pos bkg
        before_trigger = []
        for i in range(
            len(self._timeSelection.bayesian_block_times_dict[self._det]) - 1
        ):
            # check if the bayesian block is fully before the trigger range
            if (
                self._timeSelection.bayesian_block_times_dict[self._det][i + 1]
                <= start_trigger_range
            ):
                # check if the block at leas as lare as the minimum bkg block width set before
                if (
                    self._timeSelection.bayesian_block_widths_dict[self._det][i]
                    >= self._bkg_bin_min_length
                ):
                    before_trigger.append(i)
                # if the current block is to small check if the next one is big enough again and outside the trigger range
                elif (
                    self._timeSelection.bayesian_block_widths_dict[self._det][i]
                    < self._bkg_bin_min_length
                    and self._timeSelection.bayesian_block_widths_dict[self._det][i + 1]
                    >= self._bkg_bin_min_length
                ):
                    try:
                        if (
                            self._timeSelection.bayesian_block_times_dict[self._det][
                                i + 2
                            ]
                            <= start_trigger_range
                        ):
                            before_trigger.append(i)
                    except IndexError:
                        pass

        # calc neg bkg
        after_trigger = []
        for i in range(
            len(self._timeSelection.bayesian_block_times_dict[self._det]) - 1, 0, -1
        ):
            # check if the BB is fully after the trigger
            if (
                self._timeSelection.bayesian_block_times_dict[self._det][i]
                >= end_trigger_range
            ):
                try:
                    # check if the end of the block is inside the maximum time
                    if (
                        self._timeSelection.bayesian_block_times_dict[self._det][i + 1]
                        <= self._max_time
                    ):
                        # check if the block is big enough
                        if (
                            self._timeSelection.bayesian_block_widths_dict[self._det][i]
                            >= self._bkg_bin_min_length
                        ):
                            after_trigger.append(i)
                        # if current block to small check if the next one is bigger and fully outside the trigger range
                        elif (
                            self._timeSelection.bayesian_block_widths_dict[self._det][i]
                            < self._bkg_bin_min_length
                            and self._timeSelection.bayesian_block_widths_dict[
                                self._det
                            ][i - 1]
                            >= self._bkg_bin_min_length
                        ):
                            if (
                                self._timeSelection.bayesian_block_times_dict[
                                    self._det
                                ][i - 1]
                                >= end_trigger_range
                            ):
                                after_trigger.append(i)
                except IndexError:
                    pass
        try:
            # creating the return lists

            before_trigger_end = self._timeSelection.bayesian_block_times_dict[
                self._det
            ][before_trigger[-1] + 1]

            after_trigger_start = self._timeSelection.bayesian_block_times_dict[
                self._det
            ][after_trigger[-1]]

            before_trigger_bounds = [
                self._timeSelection.bayesian_block_times_dict[self._det][0],
                before_trigger_end,
            ]
            after_trigger_bounds = [after_trigger_start, self._max_time]
            return before_trigger_bounds, after_trigger_bounds

        except IndexError:
            # if the bkg selection before failed because no blocks could be selected due to constraints - weaken the constraints
            if self._bkg_bin_min_length - 1 > 0:
                self._bkg_bin_min_length -= 1
                if self._bkg_distance_trigger - 1 > 0:
                    self._bkg_distance_trigger -= 1
                print(
                    f"Conditions too hard, decreasing min length of blocks to {self._bkg_bin_min_length} and setting distance to trigger to {self._bkg_distance_trigger}"
                )
                return self._selectBackground()
            else:
                print(
                    f"Conditions still too hard - setting background without checking"
                )
                try:
                    if (
                        len(before_trigger_bounds) > 0
                        and len(after_trigger_bounds) == 0
                    ):
                        before_trigger_bounds, [
                            end_trigger_range + 50,
                            self._timeSelection.bayesian_block_times_dict[self._det][
                                -1
                            ],
                        ]
                    elif (
                        len(before_trigger_bounds) == 0
                        and len(after_trigger_bounds) > 0
                    ):
                        return [
                            self._timeSelection.bayesian_block_times_dict[self._det][0],
                            start_trigger_range - 30,
                        ], after_trigger_bounds
                    else:
                        return [
                            self._timeSelection.bayesian_block_times_dict[self._det][0],
                            start_trigger_range - 30,
                        ], [
                            end_trigger_range + 50,
                            self._timeSelection.bayesian_block_times_dict[self._det][
                                -1
                            ],
                        ]

                except UnboundLocalError:
                    return [
                        self._timeSelection.bayesian_block_times_dict[self._det][0],
                        start_trigger_range - 30,
                    ], [
                        end_trigger_range + 50,
                        self._timeSelection.bayesian_block_times_dict[self._det][-1],
                    ]

    def _polyFit(
        self, background_sel_strings=None, active_sel_string=None, det_sel=None
    ):
        """Runs the background fit by setting active times and bkg times

        Args:
            background_sel_strings (list, optional): List containing strings with start and end times of neg and pos bkg. Defaults to None.
            active_sel_string (str, optional): String with known start and stop time of active trigger time. Defaults to None.
            det_sel (str, optional): Detector name if bkg fit for just a single detector. Defaults to None.
        """

        if background_sel_strings is None:
            background_sel_strings = (
                f"{self._timeSelection._bkg_list_dict[self._det][0][0]}-{self._timeSelection._bkg_list_dict[self._det][0][1]}",
                f"{self._timeSelection._bkg_list_dict[self._det][1][0]}-{self._timeSelection._bkg_list_dict[self._det][1][1]}",
            )
            det_sel = self._det
        if active_sel_string is None:
            active_sel_string = f"{self._timeSelection.start_trigger_dict[self._det]}-{self._timeSelection.stop_trigger_dict[self._det]}"
            det_sel = self._det
        self._timeSelection.trigreader_object.set_background_selections(
            *background_sel_strings, det_sel=det_sel
        )
        self._timeSelection.trigreader_object.set_active_time_interval(
            active_sel_string, det_sel=det_sel
        )
