import numpy as np
from threeML import *
import healpy as hp
import copy
import os

class HealpixMap(object):

    def __init__(self, analysis_results, nside = 32, error=1, n_samples_point=100):

        # assuming there is only one source!
        self._point_source_name = analysis_results.optimized_model.get_point_source_name(0)

        self._nside = nside

        self._bras = analysis_results.get_variates('%s.position.ra' % self._point_source_name).samples
        self._bdecs = analysis_results.get_variates('%s.position.dec' % self._point_source_name).samples
        self._error = error
        self._n_samples_point = n_samples_point
        self._add_systematic()

        self._generate_map()

    def _add_systematic(self):
        if self._n_samples_point>0:
            self._samples_ra_new = np.array([])
            self._samples_dec_new = np.array([])
            for i in range(len(self._bras)):
                mean = [self._bras[i],self._bdecs[i]]
                samples = self._get_systematic_samples(mean)
                self._samples_ra_new = np.append(self._samples_ra_new, samples[:,0])
                self._samples_dec_new = np.append(self._samples_dec_new, samples[:,1])
                self._samples_ra_new[self._samples_ra_new>360]-=360
                self._samples_dec_new[self._samples_dec_new>90]=90
                self._samples_dec_new[self._samples_dec_new<-90]=-90
        else:
            self._samples_ra_new = self._bras
            self._samples_dec_new = self._bdecs

    def _get_systematic_samples(self, mean): #error in deg
        sample_angles = np.random.multivariate_normal([mean[0], mean[1]],
                                                      [[(self._error)**2, 0.],
                                                       [0., (self._error)**2]],
                                                      self._n_samples_point)
        return sample_angles
    
    def _generate_map(self):

        print(self.ra_healpix)
        hpix = hp.pixelfunc.ang2pix(self._nside, self.ra_healpix, self.dec, lonlat=True)

        newmap = np.zeros(hp.nside2npix(self._nside))
        for px in hpix:
            newmap[px] += 1.0 / float(self._samples_ra_new.shape[0])

        print(newmap[newmap>0])
        self._map = newmap


    def write_map(self, filename):

        hp.write_map(filename, self._map, coord='G')

    @property
    def map(self):

        return self._map

    @property
    def ra(self):

        return self._samples_ra_new

    @property
    def ra_healpix(self):

        # healpix has RA that goes from -180 to 180
        idx = (self._samples_ra_new>= 180.0)

        samples_ra_new = copy.copy(self._samples_ra_new)

        samples_ra_new[idx] = self._samples_ra_new[idx] - 360

        return samples_ra_new

    @property
    def dec(self):

        return self._samples_dec_new


def healpix_no_sys(nside=512, result_path=None, save_path=None):
    """
    Get healpix with no sys error
    """
    if os.path.exists(save_path):
        os.remove(save_path)
    results = load_analysis_results(result_path)
    healpix = HealpixMap(results, nside=nside, n_samples_point=0)
    healpix.write_map(save_path)

def healpix_with_sys(nside=512, n_samples_point=100, sat_phi=0,
                     result_path=None, save_path=None):
    """
    Get healpix with no sys error
    """
    if os.path.exists(save_path):
        os.remove(save_path)
    if sat_phi < 45 or sat_phi > 315:
        error = 1
    elif sat_phi > 45 and sat_phi < 135:
        error = 2
    elif sat_phi > 135 and sat_phi < 225:
        error = 1
    else:
        error  =2
    results = load_analysis_results(result_path)
    healpix = HealpixMap(results,
                         nside=nside,
                         error=error,
                         n_samples_point=n_samples_point)
    healpix.write_map(save_path)
