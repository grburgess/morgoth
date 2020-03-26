import collections
import os
from datetime import datetime

import pandas as pd
import astropy.io.fits as fits
import numpy as np
import yaml

from morgoth.utils.env import get_env_value
from morgoth.exceptions.custom_exceptions import *

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class ResultReader(object):

    def __init__(self, grb_name, report_type, version, result_file, trigger_file):
        self.grb_name = grb_name
        self.report_type = report_type
        self.version = version

        self._ra = None
        self._dec = None
        self._K = None
        self._alpha = None
        self._xp = None
        self._beta = None
        self._index = None
        self._xc = None

        self._ra_err = None
        self._dec_err = None
        self._K_err = None
        self._alpha_err = None
        self._xp_err = None
        self._beta_err = None
        self._index_err = None
        self._xc_err = None

        self._trigger_number = None
        self._trigger_timestamp = None
        self._data_timestamp = None
        self._most_likely = None
        self._second_most_likely = None
        self._swift = None

        # read trigger output
        self._read_trigger(trigger_file)

        # read parameter values
        self._read_fit_result(result_file)

        self._build_report()


    def _read_fit_result(self, result_file):

        with fits.open(result_file) as f:
            values = f['ANALYSIS_RESULTS'].data['VALUE']
            pos_error = f['ANALYSIS_RESULTS'].data['POSITIVE_ERROR']
            neg_error = f['ANALYSIS_RESULTS'].data['NEGATIVE_ERROR']

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

        if self.report_type == 'trigdat':
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
                self._model = 'cpl'
            except:
                self._model = 'pl'

        elif self.report_type == 'tte':
            self._model = 'band'
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
            raise UnkownReportType('The specified report type is not valid. Valid report types: (trigdat, tte)')

    def _read_trigger(self, trigger_file):

        # TODO: read file
        self._trigger_number = 123456
        self._trigger_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self._data_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self._most_likely = 'GRB 100%'
        self._second_most_likely = None
        self._swift = None

    def __repr__(self):
        """
        Examine the currently selected info as well other things.

        """
        print(f'Result Reader for {self.grb_name}')
        return self._output().to_string()

    def _output(self):

        info_dict = collections.OrderedDict()
        info_dict['ra'] = self._ra
        info_dict['ra_err']= self._ra_err
        info_dict['dec'] = self._dec
        info_dict['_dec_err']= self._dec_err
        info_dict['K'] = self._K
        info_dict['K_err']= self._K_err
        info_dict['alpha'] = self._alpha
        info_dict['alpha_err']= self._alpha_err
        info_dict['xp'] = self._xp
        info_dict['xp_err']= self._xp_err
        info_dict['beta'] = self._beta
        info_dict['beta_err']= self._beta_err
        info_dict['index'] = self._index
        info_dict['index_err']= self._index_err
        info_dict['xc'] = self._xc
        info_dict['xc_err']= self._xc_err
        info_dict['model'] = self._model


        #return pd.Series(info_dict, index=info_dict.keys())

        return pd.Series(self._report, index=self._report.keys())


    def _build_report(self):
        self._report = {

            "general": {

                "grb_name": f'{self.grb_name}',

                "version": f'{self.version}',

                "trigger_number": self._trigger_number,

                "trigger_timestamp": self._trigger_timestamp,

                "data_timestamp": self._data_timestamp,

                "localization_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),

                "most_likely": self._most_likely,

                "second_most_likely": self._second_most_likely,

                "swift": self._swift
            },

            "fit_result": {

                "model": self._model,

                "ra": self._ra,

                "ra_err": self._ra_err,

                "dec": self._dec,

                "dec_err": self._dec_err,

                "spec_K": self._K,

                "spec_K_err": self._K_err,

                "spec_index": self._index,

                "spec_index_err": self._index_err,

                "spec_xc": self._xc,

                "spec_xc_err": self._xc_err,

                "spec_alpha": self._alpha,

                "spec_alpha_err": self._alpha_err,

                "spec_xp": self._xp,

                "spec_xp_err": self._xp_err,

                "spec_beta": self._beta,

                "spec_beta_err": self._beta_err,

                "sat_phi": 15,

                "sat_theta": 15,

                "balrog_one_sig_err_circle": 15.,

                "balrog_two_sig_err_circle": 20.
            },

            "time_selection": {

                "bkg_neg_start": -15.,

                "bkg_neg_stop": 16.,

                "bkg_pos_start": 14.,

                "bkg_pos_stop": 15.,

                "active_time_start": 12.,

                "active_time_stop": 16.,

                "used_detectors": ['n1', 'n2', 'n3'],
            }
        }

    def save_result_yml(self):
        filename = f"{self.report_type}_{self.version}_fit_result.yml"
        file_path = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        with open(file_path, "w") as f:
            yaml.dump(self._report, f, default_flow_style=False)

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
