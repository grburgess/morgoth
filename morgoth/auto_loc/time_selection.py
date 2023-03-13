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
        trig_reader = TrigReader(
            self._trigdat_file, fine=self._fine, verbose=False)

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

    def __init__(self, active_time, background_time_neg,
                 background_time_pos, poly_order=-1, max_time=None,
                 fine=False):
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
            self._active_time_start = -1*float(inter[1])
            self._active_time_stop = float(inter[2])
        else:
            self._active_time_start = -1*float(inter[1])
            self._active_time_stop = -1*float(inter[3])

        inter = self._background_time_pos.split("-")
        if len(inter) == 2:
            self._bkg_pos_start = float(inter[0])
            self._bkg_pos_stop = float(inter[1])
        elif len(inter) == 3:
            self._bkg_pos_start = -1*float(inter[1])
            self._bkg_pos_stop = float(inter[2])
        else:
            self._bkg_pos_start = -1*float(inter[1])
            self._bkg_pos_stop = -1*float(inter[3])

        inter = self._background_time_neg.split("-")
        if len(inter) == 2:
            self._bkg_neg_start = float(inter[0])
            self._bkg_neg_stop = float(inter[1])
        elif len(inter) == 3:
            self._bkg_neg_start = -1*float(inter[1])
            self._bkg_neg_stop = float(inter[2])
        else:
            self._bkg_neg_start = -1*float(inter[1])
            self._bkg_neg_stop = -1*float(inter[3])


class TimeSelectionBB(TimeSelection):
    """Automatically sets active trigger time as well as neg and pos (before and after trigger) background
    """

    def __init__(self, trigdat_file, trigger_name, fine=False, gamma=0.776):
        """ Starts Timeselection

        Args:
            trigdat_file (path like): path to trigdat file
            trigger_name (str): name of the trigger
            fine (bool, optional): Use fine data binning. Defaults to False.
            gamma (float, optional): gamma value for bayesian blocks (influences number of blocks). Defaults to 0.776.
        """
        self._fine = fine
        self._trigdat_file = trigdat_file

        # Load Trigdat file
        self._trigdat_obj = TrigReader(self._trigdat_file, fine=self._fine)

        self.dets = ["n0", "n1", "n2", "n3", "n4", "n5",
                     "n6", "n7", "n8", "n9", "na", "nb", "b0", "b1"]
        self.trigger_name = trigger_name

        self._gamma = gamma

        # initialize dicts
        self._bayesian_block_edges_dict = {}
        self._bayesian_block_times_dict = {}
        self._bayesian_block_cps_dict = {}
        self._bayesian_block_widths_dict = {}

        self._start_trigger_dict = {}
        self._end_trigger_dict = {}

        self.pos_bkg_dict = {}
        self.neg_bkg_dict = {}
        self._bkg_list_dict = {}
        self._significance_dict = {}

        self._poly_order = -1

        # get relevant data from Trigdat object
        self._processTrigdat()

        # run timeselection on every detector
        self.timeselection()

        # combine most significant lightcurves
        self.fixSelections()

    @property
    def trigdat_object(self):
        return self._trigdat_obj

    @property
    def start_trigger_dict(self):
        return self._start_trigger_dict

    @property
    def end_trigger_dict(self):
        return self._end_trigger_dict

    @property
    def bayesian_block_times_dict(self):
        return self._bayesian_block_times_dict

    @property
    def bayesian_block_widths_dict(self):
        return self._bayesian_block_widths_dict

    @property
    def start_trigger(self):
        return self._active_time_start

    @property
    def end_trigger(self):
        return self._active_time_stop

    def timeselection(self, lower_trigger_bound=-4, upper_trigger_bound=50, max_trigger_length=10.24):
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

            self._calcStartStopTrigger(det)
            self.neg_bkg_dict[det] = []
            self.pos_bkg_dict[det] = []
            self._bkg_list_dict[det] = []

            # calculate the background for det
            bkgSelectorDets = BackgroundSelector(self, det)
            bkgSelectorDets.runSelector()

            # calculate the significance during the trigger time
            self._getSignificance(det)

    def fixSelections(self, significance_dets_nr=3):
        """Improves the timeselection by combining multiple detectors with the highest significance in the data

        Args:
            significance_dets_nr (int, optional): Number of detectors which will be used for combined timeselection. Defaults to 3.
        """
        significance_max = {}

        # TODO apply constraints on time for significance!
        for det, sign in self._significance_dict.items():
            # set the significance outside the trigger selection to zero to prevent "false" high significance
            sign[:self._times.index(self._start_trigger_dict[det])] = 0
            sign[self._times.index(self._end_trigger_dict[det]):] = 0
            max_det = np.max(sign)
            significance_max[det] = max_det

        # sort the significance_max dict in descending order
        significance_max = dict(
            sorted(significance_max.items(), key=lambda item: item[1], reverse=True))

        # empty list and string for combination of detectors
        obs_combined = []
        det = ""
        for i in range(significance_dets_nr):
            det += list(significance_max.keys())[i]
            obs_combined.append(
                self._cps_dets[list(significance_max.keys())[i]])

        print(
            f'The detecors {det} had the highest significance and are used to fix the selections')
        obs_combined = np.sum(obs_combined, axis=0)
        self._cps_dets[det] = obs_combined
        self._bayesianBlocks(det)

        # rebinning and timeselection
        self._bayesian_block_times_dict[det], self._bayesian_block_cps_dict[det], self._bayesian_block_widths_dict[det] = bb_binner(
            self._times, obs_combined, self._bayesian_block_edges_dict[det])

        self._calcStartStopTrigger(det)
        start_trigger = self._start_trigger_dict[det]
        end_trigger = self._end_trigger_dict[det]

        bkgSelector = BackgroundSelector(self, det)
        bkgSelector.runSelector()
        before_trigger = self.neg_bkg_dict[det]
        after_trigger = self.pos_bkg_dict[det]
        background_sel = [before_trigger, after_trigger]

        self._background_time_pos = f'{background_sel[1][0]}-{background_sel[1][1]}'
        self._background_time_neg = f'{background_sel[0][0]}-{background_sel[0][1]}'

        self._active_time = f'{start_trigger}-{end_trigger}'

        self._active_time_start = start_trigger
        self._active_time_stop = end_trigger

        self._bkg_neg_start = float(background_sel[0][0])
        self._bkg_neg_stop = float(background_sel[0][1])

        self._bkg_pos_start = float(background_sel[1][0])
        self._bkg_pos_stop = float(background_sel[1][1])

        self._max_time = float(background_sel[1][1])

        background_sel_strings = self._background_time_neg, self._background_time_pos

        bkgSelector._polyFit(
            background_sel_strings=background_sel_strings, active_sel_string=self._active_time)

    def set_max_time(self, max_time):

        self._max_time = max_time
        self.timeselection()
        self.fixSelections()

    def _processTrigdat(self):
        """ Loads trigdat data and stores times, observed cps and bin widths
        """
        # get cps and times from Trigdat object
        obs_array, _ = self._trigdat_obj.observed_and_background()
        start_times, end_times = self._trigdat_obj.tstart_tstop()
        times_dets = start_times.tolist()

        # Fix for bayesian Blocks (add an additional block with count rate zero )
        times_dets.append(end_times[-1])  # !TODO check if this is possible!
        obs_dets = []
        for i in range(len(obs_array)):
            temp = list(map(int, obs_array[i].tolist()))
            temp.append(0)
            obs_dets.append(temp)

        w_ori = [times_dets[i+1]-times_dets[i]
                 for i in range(len(times_dets)-1)]

        # set last bin length the same as the one before (approx 8s)
        w_ori.append(w_ori[-1])

        self._times = times_dets
        self._max_time = times_dets[-1]

        self._cps_dets = {}
        self._timebin_widths = {}
        for det_nr, det in enumerate(self.dets):
            self._cps_dets[det] = obs_dets[det_nr]
            self._timebin_widths[det] = w_ori[det_nr]

    def _bayesianBlocks(self, det):
        """Calculates edges of bayesian blocks and gets corresponding times, cps and widths of bins

        Args:
            det (str): Name of detector (or multiple ones)
        """
        self._bayesian_block_edges_dict[det] = bayesian_blocks(
            self._times, self._cps_dets[det], fitness="events", gamma=self._gamma)
        self._bayesian_block_times_dict[det], self._bayesian_block_cps_dict[det], self._bayesian_block_widths_dict[det] = bb_binner(
            self._times, self._cps_dets[det], self._bayesian_block_edges_dict[det])

    def _calcStartStopTrigger(self, det):
        """Calculates the start and stop time of the trigger

        Args:
            det (str): detector name
        """

        start_trigger = 0
        end_trigger = 0

        cps_temp = np.copy(self._bayesian_block_cps_dict[det])
        # setting first and last block to 0 as well as every block with a duration > 10s or if it's end is outside the set bounds
        cps_temp[-1] = 0
        cps_temp[0] = 0

        for i, val in enumerate(self._bayesian_block_widths_dict[det]):
            if cps_temp[i] != 0:
                if val > 10:
                    cps_temp[i] = 0
                else:
                    if self._bayesian_block_times_dict[det][i] < self._lower_trigger_bound or self._bayesian_block_times_dict[det][i] > self._upper_trigger_bound:
                        cps_temp[i] = 0

        id_max_cps_bb = np.argmax(cps_temp)
        # start length
        length_in = self._bayesian_block_widths_dict[det][id_max_cps_bb]
        id_l = id_max_cps_bb
        id_h = id_max_cps_bb

        while True:
            length_out, id_l, id_h = self._getNewLength(
                length_in, id_l, id_h, det)
            if length_in == length_out:
                break
            else:
                length_in = length_out

        # ToDo get Indicies and length in actual time

        start_trigger = self._bayesian_block_times_dict[det][id_l]
        end_trigger = self._bayesian_block_times_dict[det][id_h+1]

        # if the start time is still larger than end time (not sure if still needed)
        if start_trigger > end_trigger:
            start_trigger = end_trigger
            end_trigger = self._bayesian_block_times_dict[det][id_l]

        start_trigger_id, end_trigger_id = self.startStopToObsTimes(
            start_trigger=start_trigger, end_trigger=end_trigger)
        start_trigger = self._times[start_trigger_id]
        end_trigger = self._times[end_trigger_id]
        length_in = end_trigger-start_trigger
        while True:
            length_out, start_trigger_id, end_trigger_id = self._getNewLength(
                length_in, start_trigger_id, end_trigger_id, det)
            if length_in == length_out:
                break
            else:
                length_in = length_out

        self._start_trigger_dict[det] = start_trigger
        self._end_trigger_dict[det] = end_trigger
        print(
            f'Set trigger time for det {det} to {start_trigger}-{end_trigger}')

    def _getNewLength(self, length_in, id_l, id_h, det):
        """ Tries to get new length for trigger selection for bayesian blocks

        Args:
            length_in (float): current length of trigger
            id_l (int): current start id of trigger
            id_h (int): current end id of trigger
            det (str): detector name

        Returns:
            (float, int, int): new trigger length, start id, end id
        """

        # TODO this is just messy

        if id_l > id_h:
            return -length_in, id_h, id_l

        # try setting new lengths and counts, fails if index out of bounds (shouldnt, if so returning break condition)
        try:
            length_l = length_in + \
                self._bayesian_block_widths_dict[det][id_l-1]
            length_h = length_in + \
                self._bayesian_block_widths_dict[det][id_h+1]
            # counts for comparing if add block to start or stop
            counts_l = self._bayesian_block_widths_dict[det][id_l -
                                                             1] * self._bayesian_block_cps_dict[det][id_l-1]
            counts_h = self._bayesian_block_widths_dict[det][id_h +
                                                             1] * self._bayesian_block_cps_dict[det][id_h+1]
        except IndexError:
            return length_in, id_l, id_h

        # break conditions
        if length_h >= self._max_trigger_length and length_l >= self._max_trigger_length:
            return length_in, id_l, id_h

        elif length_h >= self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_l-1] < self._lower_trigger_bound:
            return length_in, id_l, id_h

        elif length_h < self._max_trigger_length and length_l >= self._max_trigger_length and self._bayesian_block_times_dict[det][id_h+1] > self._upper_trigger_bound:
            return length_in, id_l, id_h

        elif length_h < self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_h+1] > self._upper_trigger_bound and self._bayesian_block_times_dict[det][id_l-1] < self._lower_trigger_bound:
            return length_in, id_l, id_h

        # add upper block
        elif length_h < self._max_trigger_length and length_l >= self._max_trigger_length and self._bayesian_block_times_dict[det][id_h+1] < self._upper_trigger_bound:
            return length_h, id_l, id_h+1

        elif length_h < self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_l-1] < self._lower_trigger_bound and self._bayesian_block_times_dict[det][id_h+1] < self._upper_trigger_bound:
            return length_h, id_l, id_h+1

        # add lower block
        elif length_h >= self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_l-1] > self._lower_trigger_bound:
            return length_l, id_l-1, id_h

        elif length_h < self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_l-1] > self._lower_trigger_bound and self._bayesian_block_times_dict[det][id_h+1] > self._upper_trigger_bound:
            return length_l, id_l-1, id_h

        # both possible
        elif length_h < self._max_trigger_length and length_l < self._max_trigger_length and self._bayesian_block_times_dict[det][id_l-1] > self._lower_trigger_bound and self._bayesian_block_times_dict[det][id_h+1] < self._upper_trigger_bound:

            if counts_h > counts_l:
                return length_h, id_l, id_h+1

            elif counts_h < counts_l:
                return length_l, id_l-1, id_h

            elif counts_h == counts_l:

                if self._bayesian_block_times_dict[det][id_l-1] > self._lower_trigger_bound:
                    return length_l, id_l-1, id_h

                else:
                    return length_h, id_l, id_h + 1
        # should never be called but who knows
        else:
            return length_in, id_l, id_h

    def startStopToObsTimes(self, start_trigger, end_trigger):
        """Converts start and end trigger times to id of times list

        Args:
            start_trigger (float): trigger start time
            end_trigger (float): trigger end time

        Returns:
            (int, int): start id, end id
        """
        start_trigger_id = 0
        end_trigger_id = len(self._times)
        for i, t in enumerate(self._times):
            if t < start_trigger and i > start_trigger_id:
                start_trigger_id = i
            elif t > end_trigger and i < end_trigger_id:
                end_trigger_id = i
        start_trigger_id += 1
        end_trigger_id -= 1

        return start_trigger_id, end_trigger_id

    def _getSignificance(self, det):
        """Calculates the significance for a given detector

        Args:
            det (str): detector name
        """

        # obs, bkg = self._trigdat_obj.observed_and_background()
        # obs = obs[self.dets.index(det)]
        # bkg = bkg[self.dets.index(det)]
        # sig = Significance(obs, bkg)
        # self._significance_dict[det] = sig.li_and_ma()
        self._significance_dict[det] = self._trigdat_obj.time_series[det].significance_per_interval


class BackgroundSelector:
    """Class for background selection
    """

    def __init__(self, timeSelection: TimeSelectionBB, det, bkg_min_length=50, min_distance_trigger=20):
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
        """Runs the Background selection
        """
        self._timeSelection.neg_bkg_dict[self._det], self._timeSelection.pos_bkg_dict[self._det] = self._selectBackground(
        )
        self._timeSelection._bkg_list_dict[self._det] = [
            self._timeSelection.neg_bkg_dict[self._det], self._timeSelection.pos_bkg_dict[self._det]]
        self._polyFit()

    def _selectBackground(self):
        """Selects the pos and neg bkg times

        Returns:
            (list,list): (neg_bkg_bounds, pos_bkg_bounds)
        """

        start_trigger_sector = self._timeSelection.start_trigger_dict[self._det] - \
            self._bkg_distance_trigger
        end_trigger_sector = self._timeSelection.end_trigger_dict[self._det] + \
            self._bkg_distance_trigger

        before_trigger = []
        for i in range(len(self._timeSelection.bayesian_block_times_dict[self._det])-1):
            if self._timeSelection.bayesian_block_times_dict[self._det][i+1] <= start_trigger_sector:
                if self._timeSelection.bayesian_block_widths_dict[self._det][i] >= self._bkg_bin_min_length:
                    before_trigger.append(i)
                elif self._timeSelection.bayesian_block_widths_dict[self._det][i] < self._bkg_bin_min_length and self._timeSelection.bayesian_block_widths_dict[self._det][i+1] >= self._bkg_bin_min_length:
                    try:
                        if self._timeSelection.bayesian_block_times_dict[self._det][i+2] <= start_trigger_sector:
                            before_trigger.append(i)
                    except IndexError:
                        pass

        after_trigger = []
        for i in range(len(self._timeSelection.bayesian_block_times_dict[self._det])-1, 0, -1):
            if self._timeSelection.bayesian_block_times_dict[self._det][i] >= end_trigger_sector:
                try:
                    if self._timeSelection.bayesian_block_times_dict[self._det][i+1] <= self._max_time:
                        if self._timeSelection.bayesian_block_widths_dict[self._det][i] >= self._bkg_bin_min_length:
                            after_trigger.append(i)
                        elif self._timeSelection.bayesian_block_widths_dict[self._det][i] < self._bkg_bin_min_length and self._timeSelection.bayesian_block_widths_dict[self._det][i-1] >= self._bkg_bin_min_length:
                            if self._timeSelection.bayesian_block_times_dict[self._det][i-1] >= end_trigger_sector:
                                after_trigger.append(i)
                except IndexError:
                    pass
        try:
            before_trigger_end = self._timeSelection.bayesian_block_times_dict[
                self._det][before_trigger[-1]+1]

            after_trigger_start = self._timeSelection.bayesian_block_times_dict[
                self._det][after_trigger[-1]]

            # TODO apply condition only 2 consecutive smaller blocks allowed
            before_trigger_bounds = [
                self._timeSelection.bayesian_block_times_dict[self._det][0], before_trigger_end]
            after_trigger_bounds = [
                after_trigger_start, self._max_time]
            return before_trigger_bounds, after_trigger_bounds

        except IndexError:

            if self._bkg_bin_min_length - 1 > 0:
                self._bkg_bin_min_length -= 1
                if self._bkg_distance_trigger - 1 > 0:
                    self._bkg_distance_trigger -= 1
                print(
                    f'Conditions too hard, decreasing min length of blocks to {self._bkg_bin_min_length} and setting distance to trigger to {self._bkg_distance_trigger}')
                return self._selectBackground()
            else:
                print(f'Conditions still too hard - setting background without checking')
                try:
                    if len(before_trigger_bounds) > 0 and len(after_trigger_bounds) == 0:
                        before_trigger_bounds, [
                            end_trigger_sector+50, self._timeSelection.bayesian_block_times_dict[self._det][-1]]
                    elif len(before_trigger_bounds) == 0 and len(after_trigger_bounds) > 0:
                        return [self._timeSelection.bayesian_block_times_dict[self._det][0], start_trigger_sector-30], after_trigger_bounds
                    else:
                        return [self._timeSelection.bayesian_block_times_dict[self._det][0], start_trigger_sector-30], [end_trigger_sector+50, self._timeSelection.bayesian_block_times_dict[self._det][-1]]

                except UnboundLocalError:
                    return [self._timeSelection.bayesian_block_times_dict[self._det][0], start_trigger_sector-30], [end_trigger_sector+50, self._timeSelection.bayesian_block_times_dict[self._det][-1]]

    def _polyFit(self, background_sel_strings=None, active_sel_string=None, det_sel=None):
        """Runs the background fit by setting active times and bkg times

        Args:
            background_sel_strings (list, optional): List containing strings with start and end times of neg and pos bkg. Defaults to None.
            active_sel_string (str, optional): String with known start and stop time of active trigger time. Defaults to None.
            det_sel (str, optional): Detector name if bkg fit for just a single detector. Defaults to None.
        """

        if background_sel_strings is None:
            background_sel_strings = f'{self._timeSelection._bkg_list_dict[self._det][0][0]}-{self._timeSelection._bkg_list_dict[self._det][0][1]}', f'{self._timeSelection._bkg_list_dict[self._det][1][0]}-{self._timeSelection._bkg_list_dict[self._det][1][1]}'
            det_sel = self._det
        if active_sel_string is None:
            active_sel_string = f'{self._timeSelection.start_trigger_dict[self._det]}-{self._timeSelection.end_trigger_dict[self._det]}'
            det_sel = self._det
        self._timeSelection.trigdat_object.set_background_selections(
            *background_sel_strings, det_sel=det_sel)
        self._timeSelection.trigdat_object.set_active_time_interval(
            active_sel_string, det_sel=det_sel)
