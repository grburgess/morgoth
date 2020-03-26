from morgoth.auto_loc.utils.fit import MultinestFitTrigdat, MultinestFitTTE
import sys

import warnings

warnings.simplefilter('ignore')

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size > 1:
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        using_mpi = False
except:
    using_mpi = False

grb_name = sys.argv[1]
version = sys.argv[2]
bkg_fit_yaml_file = sys.argv[3]
time_selection_yaml_file = sys.argv[4]
data_type = sys.argv[5]

# get fit object

if data_type == 'trigdat':
    multinest_fit = MultinestFitTrigdat(grb_name, version, bkg_fit_yaml_file, time_selection_yaml_file)
    multinest_fit.fit()
    multinest_fit.spectrum_plot()

elif data_type == 'tte':
    multinest_fit = MultinestFitTTE(grb_name, version, bkg_fit_yaml_file, time_selection_yaml_file)
    multinest_fit.fit()
    multinest_fit.spectrum_plot()

else:
    raise AssertionError("Please use either tte or trigdat as input")
