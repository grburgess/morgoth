import numpy as np

import collections

from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike

from threeML.utils.time_interval import TimeIntervalSet

import astropy.io.fits as fits

from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.drmgen_trig import DRMGenTrig



# This is a holder of the detector names

lu = ('n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1')


class TrigReader(object):
    """
    This class reads a GBM trigdat file and performs background fitting, source selection, and plotting.
    It is also fed to the Balrog when performing localization with trigdat data.
    :param triddat_file: string that is the path to the trigdat file you wish ot read
    :param fine: optional argument to use trigdat fine resolution data. Defaults to False
    :poly_order: optional argument to set the order of the polynomial used in the background fit.
    """

    def __init__(self, trigdat_file, fine=False, time_resolved=False, verbose=True, poly_order=-1):

        # self._backgroundexists = False
        # self._sourceexists = False

        self._verbose = verbose
        self._time_resolved = time_resolved
        self._poly_order = poly_order
        # Read the trig data file and get the appropriate info

        trigdat = fits.open(trigdat_file)
        self._filename = trigdat_file
        self._out_edge_bgo = np.array(
            [
                150., 400.0, 850.0, 1500.0, 3000.0, 5500.0, 10000.0, 20000.0,
                50000.0
            ],
            dtype=np.float32)
        self._out_edge_nai = np.array(
            [3.4, 10.0, 22.0, 44.0, 95.0, 300.0, 500.0, 800.0, 2000.],
            dtype=np.float32)
        self._binwidth_bgo = self._out_edge_bgo[1:] - self._out_edge_bgo[:-1]
        self._binwidth_nai = self._out_edge_nai[1:] - self._out_edge_nai[:-1]

        # Get the times
        evntrate = "EVNTRATE"

        self._trigtime = trigdat[evntrate].header['TRIGTIME']
        self._tstart = trigdat[evntrate].data['TIME'] - self._trigtime
        self._tstop = trigdat[evntrate].data['ENDTIME'] - self._trigtime

        self._rates = trigdat[evntrate].data['RATE']

        num_times = len(self._tstart)
        self._rates = self._rates.reshape(num_times, 14, 8)

        # Obtain the positional information
        self._qauts = trigdat[evntrate].data['SCATTITD']  # [condition][0]
        self._sc_pos = trigdat[evntrate].data['EIC']  # [condition][0]

        # Get the flight software location
        self._fsw_ra = trigdat["PRIMARY"].header["RA_OBJ"]
        self._fsw_dec = trigdat["PRIMARY"].header["DEC_OBJ"]
        self._fsw_err = trigdat['PRIMARY'].header['ERR_RAD']

        # Clean up
        trigdat.close()

        # Sort out the high res times because they are dispersed with the normal
        # times.

        # The delta time in the file.
        # This routine is modeled off the procedure in RMFIT.
        myDelta = self._tstop - self._tstart
        self._tstart[myDelta < .1] = np.round(self._tstart[myDelta < .1], 4)
        self._tstop[myDelta < .1] = np.round(self._tstop[myDelta < .1], 4)

        self._tstart[~(myDelta < .1)] = np.round(self._tstart[~(myDelta < .1)],
                                                 3)
        self._tstop[~(myDelta < .1)] = np.round(self._tstop[~(myDelta < .1)],
                                                3)

        if fine:

            # Create a starting list of array indices.
            # We will dumb then ones that are not needed

            all_index = range(len(self._tstart))

            # masks for all the different delta times and
            # the mid points for the different binnings
            temp1 = myDelta < .1
            temp2 = np.logical_and(myDelta > .1, myDelta < 1.)
            temp3 = np.logical_and(myDelta > 1., myDelta < 2.)
            temp4 = myDelta > 2.
            midT1 = (self._tstart[temp1] + self._tstop[temp1]) / 2.
            midT2 = (self._tstart[temp2] + self._tstop[temp2]) / 2.
            midT3 = (self._tstart[temp3] + self._tstop[temp3]) / 2.

            # Dump any index that occurs in a lower resolution
            # binning when a finer resolution covers the interval
            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        try:

                            all_index.remove(indx)
                        except:
                            pass

            for indx in np.where(temp3)[0]:
                for x in midT2:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        try:

                            all_index.remove(indx)

                        except:
                            pass
            for indx in np.where(temp4)[0]:
                for x in midT3:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        try:

                            all_index.remove(indx)
                        except:
                            pass

            all_index = np.array(all_index)
        else:

            # Just deal with the first level of fine data
            all_index = np.where(myDelta > 1.)[0].tolist()

            temp1 = np.logical_and(myDelta > 1., myDelta < 2.)
            temp2 = myDelta > 2.
            midT1 = (self._tstart[temp1] + self._tstop[temp1]) / 2.

            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if self._tstart[indx] < x < self._tstop[indx]:

                        try:

                            all_index.remove(indx)

                        except:
                            pass

            all_index = np.array(all_index)

        # Now dump the indices we do not need
        self._tstart = self._tstart[all_index]
        self._tstop = self._tstop[all_index]
        self._qauts = self._qauts[all_index]
        self._sc_pos = self._sc_pos[all_index]
        self._rates = self._rates[all_index, :, :]

        # Now we need to sort because GBM may not have done this!

        sort_mask = np.argsort(self._tstart)
        self._tstart = self._tstart[sort_mask]
        self._tstop = self._tstop[sort_mask]
        self._qauts = self._qauts[sort_mask]
        self._sc_pos = self._sc_pos[sort_mask]
        self._rates = self._rates[sort_mask, :, :]

        self._time_intervals = TimeIntervalSet.from_starts_and_stops(
            self._tstart, self._tstop)

        # self._pos_interp = PositionInterpolator(trigdat=trigdat_file)

        self._create_timeseries()

    def _create_timeseries(self):
        """
        create all the time series for each detector
        :return: None
        """

        self._time_series = collections.OrderedDict()

        for det_num in range(14):

            # detectors are arranged [time,det,channel]

            # for now just keep the normal exposure

            # we will create binned spectra for each time slice


            drm_gen = DRMGenTrig(
                self._qauts,
                self._sc_pos,
                det_num,  # det number
                tstart=self._tstart,
                tstop=self._tstop,
                mat_type=2,
                time=0)

            # we will use a single response for each detector

            tmp_drm = BALROG_DRM(drm_gen, 0, 0)

            # extract the counts

            counts = self._rates[:, det_num, :] * self._time_intervals.widths.reshape((len(self._time_intervals), 1))

            # now create a binned spectrum for each interval

            binned_spectrum_list = []

            for c, start, stop in zip(counts, self._tstart, self._tstop):
                binned_spectrum_list.append(
                    BinnedSpectrumWithDispersion(
                        counts=c,
                        exposure=stop - start,
                        response=tmp_drm,
                        tstart=start,
                        tstop=stop))

            # make a binned spectrum set

            bss = BinnedSpectrumSet(
                binned_spectrum_list,
                reference_time=0.,
                time_intervals=self._time_intervals)

            # convert that set to a series

            bss2 = BinnedSpectrumSeries(bss, first_channel=0)

            # now we need to get the name of the detector

            name = lu[det_num]

            # create a time series builder which can produce plugins

            tsb = TimeSeriesBuilder(name, bss2, response=tmp_drm, verbose=self._verbose, poly_order=self._poly_order)

            # attach that to the full list

            self._time_series[name] = tsb

    def view_lightcurve(self, start=-30, stop=30, return_plots=False):
        """                                                                                                                                                                                                                                                                    
        view the lightcurves of all detectors and if the variable return_plots is True returns the plots                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        :param start: start time                                                                                                                                                                                                                                               
        :param stop: stop time                                                                                                                                                                                                                                                 
        :param return_plots: True if the created plots should be returned

        :return:                                                                                                                                                                                                                                                               
        """
        plots = []
        for name, det in self._time_series.iteritems():

            #try because sometimes there is no data for some dets in the trigdat files                                                                                                                                                                                         

            try:
                fig = det.view_lightcurve(start, stop)
                fig.get_axes()[0].set_title(name)
                if return_plots:
                    plots.append([name,fig])
            except:
                print('Could not create a lightcurve for detector {}'.format(name))
        if return_plots:
            return plots

        

    def set_background_selections(self, *intervals):
        """
        set the background selections for all detection

        :param intervals: str of intervals
        :return:
        """
        for name, det in self._time_series.iteritems():
            det.set_background_interval(*intervals, unbinned=False)

    def set_active_time_interval(self, *intervals):
        """
        set the selection for all intervals
        :param intervals:
        :return:
        """
        for name, det in self._time_series.iteritems():
            det.set_active_time_interval(*intervals)

    def to_plugin(self, *detectors):
        """

        convert the series to a BALROGLike plugin

        :param detectors: detectors to use
        :return:
        """

        data = []

        for det in detectors:
            # first create a DSL

            speclike = self._time_series[det].to_spectrumlike()

            # then we convert to BL

            time = 0.5 * (
                self._time_series[det].tstart + self._time_series[det].tstop)

            balrog_like = BALROGLike.from_spectrumlike(speclike, time=time)

            balrog_like.set_active_measurements('c1-c6')

            data.append(balrog_like)

        return data

    def counts_and_background(self, time_series_builder):
        """                                                                                                                                                                                                                                                                    
        Method that returns the observed rate and the rate of the poly bkg fit                                                                                                                                                                                                 
        :return: returns the observed rate and bkg rate for one detector for all time_bins                                                                                                                                                                                     
        """
        start=-1000
        stop=1000
        time_series = time_series_builder.time_series
        poly_fit_exists = time_series.poly_fit_exists
        binned_spectrum_set = time_series.binned_spectrum_set
        counts=[]
        width=[]
        bins = binned_spectrum_set.time_intervals.containing_interval(start, stop)
        for bin in bins:
            counts.append(time_series.counts_over_interval(bin.start_time, bin.stop_time) )
            width.append(bin.duration)
        counts = np.array(counts)
        width = np.array(width)
        rates_observed = counts/width

        if poly_fit_exists:
            polynomials = time_series.polynomials

            bkg = []
            for j, tb in enumerate(bins):
                tmpbkg = 0.
                for poly in polynomials:
                    tmpbkg += poly.integral(tb.start_time, tb.stop_time)

                bkg.append(tmpbkg / width[j])

        else:

            bkg = None

        rates_observed = np.array(rates_observed)
        bkg = np.array(bkg)
        return rates_observed, bkg

    def observed_and_background(self):
        """                                                                                                                                                                                                                                                                    
        Method that returns the observed rate and the rate calculated with the bkg fit. Needs the method counts_and_background in 3ML.                                                                                                                                         
        Needed for the automatic localisation script to identify the active time and bkg times.                                                                                                                                                                                
        :return: returns an list with an array in which the observed rate for all time_bins is saved for every det, same for bkg fit                                                                                                                                           
        """
        observed_rate_all = []
        background_rate_all = []
        for name, det in self._time_series.iteritems():
            observed_rate, bkg_rate = self.counts_and_background(det)
            observed_rate_all.append(observed_rate)
            background_rate_all.append(bkg_rate)

        return observed_rate_all, background_rate_all

    def tstart_tstop(self):
        """                                                                                                                                                                                                                                                                    
        :return: start and stops time of bins in trigdata                                                                                                                                                                                                                      
        """
        return self._tstart,self._tstop

    def quats_sc_time_burst(self):
        """                                                                                                                                                                                                                                                                    
        :return: returns the quat, the sc pos and the time of the trigger                                                                                                                                                                                                      
        """
        i=0
        while i<len(self._qauts):
            if self._tstart[i]>0:
                return self._qauts[i], self._sc_pos[i], self._trigtime
            i+=1
