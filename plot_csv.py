import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from DMI_model import *
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['figure.figsize'] = (20,20)
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = '/work/projects/tmqs_projects/data_SkL/out/'
sstr = '*chi_64*Lx_251*finite.csv'
fns = np.sort(find_files(sstr, dir))

out_dir = 'plots/'
os.makedirs(out_dir, exist_ok=True)

for fn in fns:
    df = pd.read_csv(fn)

    # df['\\phi'] = np.arctan2(df['S_x'], df['S_y'])

    print('CSV loaded')

    mkr = 'H'
    ms = 600
    ms = 3

    fig, ax = plt.subplots(1,1)
    imag = ax.scatter(df['x'], df['y'], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=df['S'])
    ax.quiver(df['x'], df['y'], df['S_x'], df['S_y'], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
    ax.set_aspect('equal')
    ax.set_title('$\\langle \\vec S_{i} \\rangle$')

    fig.colorbar(imag)
    ax.axis('off')

    fn_repl = fn.replace(dir, out_dir).replace('.csv', '.jpg')
    plt.tight_layout()
    plt.savefig(fn_repl, dpi=1200, bbox_inches='tight')
    plt.close()
    print(fn_repl)