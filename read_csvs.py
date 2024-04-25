import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
from aux.plot_lobs import plot_lobs
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
aps_fs = 3+3/8
gr = 1.618
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (aps_fs,aps_fs/gr)
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out/'
sstr = '*chi_64*finite.csv'
fns = np.sort(find_files(sstr, dir))

out_dir = 'plots/'
os.makedirs(out_dir, exist_ok=True)

for fn in fns:
    df = pd.read_csv(fn)

    # df['\\phi'] = np.arctan2(df['S_x'], df['S_y'])

    print('CSV loaded')

    fn_repl = fn.replace(dir, out_dir).replace('.h5', '.jpg')
    plot_lobs(df, fn_repl)
    print(fn_repl)

    fig, ax = plt.subplots(1,1)
    ys = np.unique(df['y'])
    for ny in ys:
        df_x = df[abs(df['y']-ny)<1e-4]
        ax.plot(df_x['x'], df_x['vNEE'])
    plt.tight_layout()
    fn_repl = fn.replace(dir, out_dir).replace('.csv', 'EE.jpg')
    plt.savefig(fn_repl, dpi=1200, bbox_inches='tight')
    plt.close()
    print(fn_repl)