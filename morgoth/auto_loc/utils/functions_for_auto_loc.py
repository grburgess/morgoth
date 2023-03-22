import numpy as np
import scipy
from numpy import sqrt
from scipy.special import erfinv


class PoissonResiduals(object):
    """
    This class implements a way to compute residuals for a Poisson distribution mapping them to residuals of a standard
    normal distribution. The probability of obtaining the observed counts given the expected one is computed, and then
    transformed "in unit of sigma", i.e., the sigma value corresponding to that probability is computed.
    The algorithm implemented here uses different branches so that it is fairly accurate between -36 and +36 sigma.
    NOTE: if the expected number of counts is not very high, then the Poisson distribution is skewed and so the
    probability of obtaining a downward fluctuation at a given sigma level is not the same as obtaining the same
    fluctuation in the upward direction. Therefore, the distribution of residuals is *not* expected to be symmetric
    in that case. The sigma level at which this effect is visible depends strongly on the expected number of counts.
    Under normal circumstances residuals are expected to be a few sigma at most, in which case the effect becomes
    important for expected number of counts <~ 15-20.
    """

    # Putting these here make them part of the *class*, not the instance, i.e., they are created
    # only once when the module is imported, and then are referred to by any instance of the class

    # These are lookup tables for the significance from a Poisson distribution when the
    # probability is very low so that the normal computation is not possible due to
    # the finite numerical precision of the computer

    _x = np.logspace(np.log10(5), np.log10(36), 1000)
    _logy = np.log10(scipy.stats.norm.sf(_x))

    # Make the interpolator here so we do it only once. Also use ext=3 so that the interpolation
    # will return the maximum value instead of extrapolating

    _interpolator = scipy.interpolate.InterpolatedUnivariateSpline(
        _logy[::-1], _x[::-1], k=1, ext=3
    )

    def __init__(self, Non, Noff, alpha=1.0):

        assert alpha > 0 and alpha <= 1, "alpha was %f" % alpha

        self.Non = np.array(Non, dtype=float, ndmin=1)

        self.Noff = np.array(Noff, dtype=float, ndmin=1)

        self.alpha = float(alpha)

        self.expected = self.alpha * self.Noff

        self.net = self.Non - self.expected

        # This is the minimum difference between 1 and the next representable floating point number
        self._epsilon = np.finfo(float).eps

    def significance_one_side(self):

        # For the points where Non > expected, we need to use the survival function
        # sf(x) = 1 - cdf, which can go do very low numbers
        # Instead, for points where Non < expected, we need to use the cdf which allows
        # to go to very low numbers in that directions

        idx = self.Non >= self.expected

        out = np.zeros_like(self.Non)

        if np.sum(idx) > 0:
            out[idx] = self._using_sf(self.Non[idx], self.expected[idx])

        if np.sum(~idx) > 0:
            out[~idx] = self._using_cdf(self.Non[~idx], self.expected[~idx])

        return out

    def _using_sf(self, x, exp):

        sf = scipy.stats.poisson.sf(x, exp)

        # print(sf)

        # return erfinv(2 * sf) * sqrt(2)

        return scipy.stats.norm.isf(sf)

    def _using_cdf(self, x, exp):

        # Get the value of the cumulative probability function, instead of the survival function (1 - cdf),
        # because for extreme values sf(x) = 1 - cdf(x) = 1 due to numerical precision problems

        cdf = scipy.stats.poisson.cdf(x, exp)

        # print(cdf)

        out = np.zeros_like(x)

        idx = cdf >= 2 * self._epsilon

        # We can do a direct computation, because the numerical precision is sufficient
        # for this computation, as -sf = cdf - 1 is a representable number

        out[idx] = erfinv(2 * cdf[idx] - 1) * sqrt(2)

        # We use a lookup table with interpolation because the numerical precision would not
        # be sufficient to make the computation

        out[~idx] = -1 * self._interpolator(np.log10(cdf[~idx]))

        return out


class Significance(object):
    """
    Implements equations in Li&Ma 1983
    """

    def __init__(self, Non, Noff, alpha=1):
        assert alpha > 0 and alpha <= 1, "alpha was %f" % alpha

        self.Non = np.array(Non, dtype=float, ndmin=1)

        self.Noff = np.array(Noff, dtype=float, ndmin=1)

        self.alpha = float(alpha)

        self.expected = self.alpha * self.Noff

        self.net = self.Non - self.expected

    def known_background(self):
        """
        Compute the significance under the hypothesis that there is no uncertainty in the background. In other words,
        compute the probability of obtaining the observed counts given the expected counts from the background, then
        transform it in sigma.
        NOTE: this is reliable for expected counts >~10-15 if the significance is not very high. The higher the
        expected counts, the more reliable the significance estimation. As rule of thumb, you need at least 25 counts
        to have reliable estimates up to 5 sigma.
        NOTE 2: if you use to compute residuals in units of sigma, you should not expected them to be symmetrically
        distributed around 0 unless the expected number of counts is high enough for all bins (>~15). This is due to
        the fact that the Poisson distribution is very skewed at low counts.
        :return: significance vector
        """

        # Poisson probability of obtaining Non given Noff * alpha, in sigma units

        poisson_probability = PoissonResiduals(
            self.Non, self.Noff, self.alpha
        ).significance_one_side()

        return poisson_probability


def time_with_less_sigma(residuals, tstart, tstop, sigma_lim):
    j = 0
    time_intervals_all = []
    tstart_save = tstart
    tstop_save = tstop
    while j < len(residuals):
        tstart = tstart_save
        tstop = tstop_save
        i = 0
        # get the indices of the bins that are above the threshold
        index_del = []
        while i < len(residuals[j]):
            if residuals[j][i] > sigma_lim:
                index_del.append(i)
            i += 1
        # get the indices of single bins that are above threshold => leave them out later on

        i = 0
        while i < len(index_del):
            if i == 0 and len(index_del) > 1:
                if index_del[i + 1] != index_del[i] + 1:
                    del index_del[i]
                else:
                    i += 1
            elif i == len(index_del) - 1 and len(index_del) > 1:
                if index_del[i - 1] != index_del[i] - 1:
                    del index_del[i]
                else:
                    i += 1
            elif len(index_del) > 2:
                if (
                    index_del[i - 1] != index_del[i] - 1
                    and index_del[i + 1] != index_del[i] + 1
                ):
                    del index_del[i]
                else:
                    i += 1
            else:
                i += 1
        if len(index_del) == 1:
            index_del = []

        tstart = tstart.tolist()
        tstop = tstop.tolist()
        # fill the missing time
        i = 0
        tstartadd = []
        tstopadd = []
        while i < len(tstart) - 1:
            if tstart[i + 1] > tstop[i] + 0.1:
                tstartadd.append(tstop[i])
                tstopadd.append(tstart[i + 1])
            i += 1
        # delete the bins that are above the threshold
        sub = 0
        for i in index_del:
            del tstart[i - sub]
            del tstop[i - sub]
            sub += 1
        tstart = sorted(tstart + tstartadd)
        tstop = sorted(tstop + tstopadd)
        time_bins = np.vstack((tstart, tstop)).T
        time_bins = time_bins.tolist()
        # delete time_bins outside of -150-150
        i = 0
        while i < len(time_bins):
            if time_bins[i][0] < -150 or time_bins[i][0] > 150:
                del time_bins[i]
            else:
                i += 1
        # merge time_bins that are less than 0.2 sec seperated
        i = 0
        while i < len(time_bins) - 1:
            if len(time_bins) == 1:
                not_whole_time = False
                break
            if time_bins[i + 1][0] - time_bins[i][1] < 0.2:
                time_bins[i] = [time_bins[i][0], time_bins[i + 1][1]]
                del time_bins[i + 1]
            else:
                i += 1
        # delete time selections that are shorter than 2 seconds
        i = 0
        while i < len(time_bins):
            if time_bins[i][1] - time_bins[i][0] < 2:
                del time_bins[i]
            else:
                i += 1

        # only use the result when it is not one time section from the beginning to the end
        if len(time_bins) > 1:
            time_intervals_all.append(time_bins)
        j += 1

    return time_intervals_all


def new_intervals(time_intervals_all):
    # get the largest time section out of this
    sr_small_max = 0
    sr_large_min = 0
    i = 0
    while i < len(time_intervals_all):
        if time_intervals_all[i][0][1] < sr_small_max:
            sr_small_max = time_intervals_all[i][0][1]
        if time_intervals_all[i][-1][0] > sr_large_min:
            sr_large_min = time_intervals_all[i][-1][0]
        i += 1
    # choose max_time (time up to which the bkg selection is made) to be 150 seconds or if the lower boundary for
    # bkg selection after the burst is to close such that we use at least a period of 50 seconds
    max_time = 150
    # end_of_active = sr_large_min
    if sr_large_min < 10:
        sr_large_min = 50
    if sr_large_min < 25:
        sr_large_min = 75
    else:
        sr_large_min = 1.5 * sr_large_min
    if sr_large_min > 100:
        max_time = sr_large_min + 50
    end_of_active = sr_large_min
    # new bkg selection
    print("-150-" + str(sr_small_max - 20))
    print(str(sr_large_min) + "-" + str(max_time))
    # new background selection: from -100 to -20 sec from the above defined min. value and from the above
    # defined max value (to make sure to not be in the tail of the burst) to max_time
    return sr_large_min, sr_small_max, max_time, end_of_active


def get_new_intervals(sigma_lim, trig_reader):
    # get data and bkg rate for all bins
    observed, background = trig_reader.observed_and_background()
    residuals = []
    i = 0
    while i < len(observed):
        significance_calc = Significance(observed[i], background[i])
        residuals.append(significance_calc.known_background())
        i += 1
    # get the start and stop times for all time bins
    tstart, tstop = trig_reader.tstart_tstop()
    # get the times for which the difference between data and previous bkg is less then a threshold sigma
    # this is done for each detector (ignore when only one time bin is above)
    time_intervals_all = time_with_less_sigma(
        residuals, tstart, tstop, sigma_lim)
    # get new intervals out of the new time_intervals_all
    sr_large_min, sr_small_max, max_time, end_of_active = new_intervals(
        time_intervals_all
    )
    # define the new selection

    new_bkg_neg_start = -150
    new_bkg_neg_stop = sr_small_max - 20

    new_bkg_pos_start = sr_large_min
    new_bkg_pos_stop = max_time

    # define active time + cut it down to 15 sec if it is too long
    # if end_of_active - sr_small_max < 2:
    #    active_time = str(sr_small_max) + '-' + str(sr_small_max + 2)
    # else:
    active_time_start, active_time_stop = active_time_selection(
        observed, background, sr_small_max, end_of_active, tstart, tstop
    )

    return (
        new_bkg_neg_start,
        new_bkg_neg_stop,
        new_bkg_pos_start,
        new_bkg_pos_stop,
        active_time_start,
        active_time_stop,
        max_time,
    )


def active_time_selection(
    observed, background, sr_small_max, end_of_active, tstart, tstop
):
    observed = np.sum(observed, axis=0)
    background = np.sum(background, axis=0)
    rate = observed - background
    sr_small_max -= 5
    i = 0
    found_low = False
    found_high = False
    print("sr_small_max: {}, end_of_active:{}".format(
        sr_small_max, end_of_active))
    while i < len(tstart) - 1:
        if tstart[i] > sr_small_max and not found_low:
            low_index = i
            found_low = True
        if tstart[i + 1] > end_of_active and not found_high:
            high_index = i
            found_high = True
        i += 1
    if low_index < high_index:
        max_index = low_index + np.argmax(rate[low_index:high_index])
        print("max_index {}".format(max_index))

    else:
        max_index = low_index
        high_index = low_index
    index_list = []
    index = max_index - 1
    while index >= low_index:
        if -tstart[index] + tstart[max_index] < 10 and rate[index] > 0:
            if rate[index - 1] > rate[index]:
                test1 = False
            else:
                test1 = True
            if rate[index - 2] > rate[index]:
                test2 = False
            else:
                test2 = True
            if rate[index - 5] > rate[index]:
                test3 = False
            else:
                test3 = True

            if test1 or test2 or test3:
                index_list.append(index)
                index -= 1
            else:
                index = 0
        else:
            index = 0
        if (
            rate[index] < (rate[max_index]) / 10
            and rate[index - 1] < (rate[max_index]) / 10
        ):
            index = 0

    index = max_index + 1
    while index <= high_index:
        if tstart[index] - tstart[max_index] < 10 and rate[index] > 0:

            if rate[index + 1] > rate[index]:
                test1 = False
            else:
                test1 = True
            if rate[index + 2] > rate[index]:
                test2 = False
            else:
                test2 = True
            if rate[index + 3] > rate[index]:
                test3 = False
            else:
                test3 = True
            if test1 or test2 or test3:
                index_list.append(index)
                index += 1
            else:
                index = 100 * high_index

            if index <= high_index:

                if (
                    rate[index] < (rate[max_index]) / 10
                    and rate[index + 1] < (rate[max_index]) / 10
                ):
                    index = 100 * high_index
        else:
            index = 100 * high_index

    if len(index_list) > 0:
        if (
            tstop[index_list[np.argmax(index_list)]]
            - tstart[index_list[np.argmin(index_list)]]
            < 10
        ):
            active_time_start = tstart[index_list[np.argmin(index_list)]]
            active_time_stop = tstop[index_list[np.argmax(index_list)]]

        elif (
            tstop[index_list[np.argmax(index_list)]] - tstart[max_index] > 5
            and tstart[max_index] - tstart[index_list[np.argmin(index_list)]] > 5
        ):
            active_time_start = tstart[max_index] - 5
            active_time_stop = tstart[max_index] + 5

        elif tstop[index_list[np.argmax(index_list)]] - tstart[max_index] < 5:
            active_time_start = tstart[max_index] - (
                10 - (tstop[index_list[np.argmax(index_list)]] -
                      tstart[max_index])
            )
            active_time_stop = tstop[index_list[np.argmax(index_list)]]

        else:
            active_time_start = tstart[index_list[np.argmin(index_list)]]
            active_time_stop = tstart[max_index] + (
                10 - (tstart[max_index] -
                      tstart[index_list[np.argmin(index_list)]])
            )

    else:
        active_time_start = tstart[max_index]
        active_time_stop = tstop[max_index]

    print(f"Active Time: {active_time_start}-{active_time_stop}")
    return float(active_time_start), float(active_time_stop)


def bb_binner(t, x, edges):
    """bins x- and t-values into blocks with given edges (BB = Bayesian Blocks)

    Args:
        t (array like): times of measurements
        x (array like): measurments
        edges (array like): edges of bayesian blocks

    Returns:
        list: bin start times, bin values, bin widths
    """
    bb_w = [edges[i+1]-edges[i]
            for i in range(len(edges)-1)]  # set width of BB
    bb_t = edges[:-1]  # Times of BB
    bb_x = []  # Count Rate of BB

    i = 0  # iterator over timeseries
    for edge_id, edge in enumerate(bb_t):
        avg = []    # list containing count rates over which will be averaged
        # list with weights for averaging (= length of time interval)
        weights = []

        # iterate over timeseries and append avg and weights
        while edges[edge_id+1] > t[i+1]:
            avg.append(x[i])
            weights.append(t[i+1]-t[i])
            i += 1

        # if edge is in between two time points set weight accordingly
        if edges[edge_id+1] <= t[i+1] and edges[edge_id+1] > t[i]:
            avg.append(x[i])
            weights.append(edges[edge_id+1]-t[i])
        # treat first block seperatly because it sucks
        elif edge == t[0] and edges[edge_id+1] < t[1]:
            avg.append(x[0])
            weights.append(edges[edge_id+1]-t[0])
        bb_x.append(np.average(avg, axis=0, weights=weights))

    return bb_t, bb_x, bb_w
