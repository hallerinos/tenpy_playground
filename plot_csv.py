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
mpl.rcParams['figure.figsize'] = (20,20)
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out/'
fn = f'{dir}dmrg_chi_512_Bz_-0.7700_Lx_251_Ly_9_bc_finite.csv'

out_dir = 'plots/'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(fn)

print('CSV loaded')

mkr = 'H'
ms = 30

fig, ax = plt.subplots(1,1)
imag = ax.scatter(df['x'], df['y'], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=df['S'], vmin=min(df['S']), vmax=max(df['S']))
ax.quiver(df['x'], df['y'], df['S_x'], df['S_y'], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
ax.set_aspect('equal')
ax.set_title('$\\langle \\vec S_{i} \\rangle$')

fig.colorbar(imag)
ax.axis('off')

fn_repl = fn.replace(dir, out_dir).replace('.csv', '.jpg')
plt.tight_layout()
plt.savefig(fn_repl, dpi=300, bbox_inches='tight')
plt.close()
print(fn_repl)