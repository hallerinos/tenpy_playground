import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3.0+3.0/8.0),(3.0+3.0/8.0))
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out'
chis = ['64', '128', '256', '512']
chis = ['128']
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

plot_all = True

maxlen = np.max([len(f) for f in fnss])
av_Mz = np.zeros((len(fnss),2,maxlen))
for (idfns,fns) in enumerate(fnss):
    for (idfn,fn) in enumerate(fns):
        print(fn)
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                sws = hdf5_io.load_from_hdf5(f, "/sweep_stats")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except:
            print(f'file: {fn} not readable')
            continue

        fig, axs = plt.subplots(2,2)
        axs = axs.ravel()
        ax = axs[0]
        imag = ax.plot(sws['E'][:-2]-sws['E'][-1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax = axs[1]
        imag = ax.plot(sws['max_S'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax = axs[2]
        imag = ax.plot(sws['norm_err'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax = axs[3]
        imag = ax.plot(sws['max_E_trunc'])
        ax.set_xscale('log')
        ax.set_yscale('log')

        fn_repl = fn.replace('.h5', '')
        plt.savefig(f'{fn_repl}_conv.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
        plt.close()
        exit()