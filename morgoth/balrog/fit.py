from auto_loc.fit.fit import fit
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

fit_running_path = sys.argv[1]
try:
    poly_order = int(sys.argv[2])
except:
    poly_order = -1
try:
    model = sys.argv[3]
except:
    model=None
try:
    ctime = sys.argv[4]
except:
    ctime = False
try:
    tte = sys.argv[5]
except:
    tte = False
# get fit object
if using_mpi:
    success_init = True
    try:
        fit = fit(fit_running_path, fixed_poly_order=poly_order, model=model, ctime=ctime, tte=tte)

    except Exception as e:
        print(e)
        success_init = False
    success_init_g = comm.gather(success_init, root=0)
    failed = []
    if rank == 0:
        print('Here!')
        print(success_init_g)
        for i, entry in enumerate(success_init_g):
            if not entry:
                failed.append(i)
        if len(failed) > 0:
            print('Init failed in these ranks: {}'.format(failed))
    failed = comm.bcast(failed, root=0)
    if len(failed) == 0:
        fit.fit()
        fit.spectrum_plot()
else:
    fit = fit(fit_running_path, fixed_poly_order=poly_order, model=model, ctime=ctime)
    # do the fit
    fit.fit()
    # save spectrum plot
    fit.spectrum_plot()
