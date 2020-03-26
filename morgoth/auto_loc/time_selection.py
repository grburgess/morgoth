import os
import yaml

from trigdat_reader import *
from auto_loc.utils.general.FunctionsForAutoLoc import *

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelection(object):

    def __init__(self, grb_name, version):
        """
        Object for time selection of a GRB event.
        :param grb_name: Name of GRB
        :param version: Version of trigdat data file
        """
        self._grb_name = grb_name
        self._version = version

        self.trigdata_time_selection()
        
    def trigdata_time_selection(self):
        """
        Function to calcuate the time selection for a given trigger event. This is done iterative.
        :return:
        """
        trigger_path = os.path.join(base_dir, self._grb_name, f"glg_trigdat_all_bn{self._grb_name[3:]}_{self._version}.fit")
        
        trig_reader = TrigReader(trigger_path,
                                 fine=False,
                                 verbose=False)

        # Inital bkg and active time selection - We will change this recursivly to explain most of
        # the bkg by a polynomial
        trig_reader.set_active_time_interval('0-0')
        trig_reader.set_background_selections('-150-0', '0-150')

        # In several steps cut away data from background section that are at least in one
        # detector above the sigma_lim value
        
        sigma_lims = [100, 50, 30, 10, 7, 5, 3]
        for sigma_lim in sigma_lims:
            new_background_selection_neg, new_background_selection_pos, active_time, max_time = \
                newIntervalWholeCalc(sigma_lim, trig_reader)
            # set active_time new and set background selection new => new background fit with this selection                                                                                                                                                               
            trig_reader.set_active_time_interval(active_time)
            trig_reader.set_background_selections(new_background_selection_neg, new_background_selection_pos)
            
        self._background_time_neg = new_background_selection_neg
        self._background_time_pos = new_background_selection_pos
        self._active_time = active_time
        self._max_time = max_time

    def save_yaml(self, path):
        """
        Save the automatic time selection in a yaml file
        :param path: Path where to save the yaml file
        :return:
        """
        time_select = {}
        
        time_select['Active_Time'] = self._active_time
        time_select['Background_Time'] = {'Time_Before': self._background_time_neg,
                                          'Time_After': self._background_time_pos}
        time_select['Max_Time'] = self._max_time

        # Poly_Order entry with -1 (default). But we need this entry in the
        # yaml file to give us the possibility to alter it if we see that it screws up one of
        # the det fits
        time_select['Poly_Order'] = -1
        
        with open(path, "w") as outfile: 
            yaml.dump(time_select, outfile)
        
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

        assert string==None or (tstart==None and tstop==None), 'Only use string definition or start and stop time definition!'
        assert string!=None or (tstart!=None and tstop!=None), 'String definition and start and stop time are both set to None!'
        
        if string is not None:
            self._background_time_pos = string
        else:
            self._background_time_pos = '{}-{}'.format(tstart,tstop)

    def set_background_time_neg(self, tstart=None, tstop=None, string=None):

        assert string==None or (tstart==None and tstop==None), 'Only use string definition or start and stop time definition!'
        assert string!=None or (tstart!=None and tstop!=None), 'String definition and start and stop time are both set to None!'
        
        if string is not None:
            self._background_time_neg = string
        else:
            self._background_time_neg = f'{tstart}-{tstop}'

    def set_active_time(self, tstart=None, tstop=None, string=None):

        assert string==None or (tstart==None and tstop==None), 'Only use string definition or start and stop time definition!'
        assert string!=None or (tstart!=None and tstop!=None), 'String definition and start and stop time are both set to None!'

        if string is not None:
            self._active_time = string
        else:
            self._active_time = f'{tstart}-{tstop}'

    def set_max_time(self, max_time):

        self.max_time = max_time
