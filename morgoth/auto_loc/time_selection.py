import os

import yaml
from morgoth.utils.trig_reader import TrigReader

from morgoth.auto_loc.utils.functions_for_auto_loc import *

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelection(object):
    def __init__(self, grb_name, version, trigdat_file):
        """
        Object for time selection of a GRB event.
        :param grb_name: Name of GRB
        :param version: Version of trigdat data file
        """
        self._grb_name = grb_name
        self._version = version
        self._trigdat_file = trigdat_file

        self.trigdata_time_selection()

    def trigdata_time_selection(self):
        """
        Function to calcuate the time selection for a given trigger event. This is done iterative.
        :return:
        """
        trig_reader = TrigReader(self._trigdat_file, fine=False, verbose=False)

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
            "poly_order": -1,
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
