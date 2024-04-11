import h5py
from tenpy.tools import hdf5_io
import matplotlib.pyplot as plt
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files

import pandas as pd

dir = '.'
sstr = '*results_B*].h5'
fns = find_files(sstr, dir)

for fn in fns:
    with h5py.File(fn, 'r') as f:
        # or for partial reading:
        mms = hdf5_io.load_from_hdf5(f, "/measurements")
        sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")

    # print(sim['model_params'])
    df = mms['lobs']
    df = df[1]

    fig, ax = plt.subplots(1,1)
    ax.scatter(df[:,0], df[:,1], marker='H', s=230, cmap='RdBu_r', c=df[:,4], vmin=-0.5, vmax=0.5)
    ax.quiver(df[:,0], df[:,1], df[:,2], df[:,3], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
    ax.set_aspect('equal')

    mx = np.asarray([np.min(df[:,0]),np.max(df[:,0])])
    my = np.asarray([np.min(df[:,1]),np.max(df[:,1])])
    ax.set_xlim(1.25*mx)
    ax.set_ylim(1.25*my)
    ax.axis('off')
    fn_repl = fn.replace('.h5', '')
    plt.savefig(f'{fn_repl}test.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.close()
