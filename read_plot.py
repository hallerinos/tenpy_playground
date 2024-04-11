import h5py
from tenpy.tools import hdf5_io
import matplotlib.pyplot as plt
import numpy as np
from tenpy.models import lattice

import pandas as pd

fn = 'results_B_[0.0, 0.0, -0.9].h5'

with h5py.File(fn, 'r') as f:
    # or for partial reading:
    mms = hdf5_io.load_from_hdf5(f, "/measurements")
    sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")

# print(sim['model_params'])
df = mms['lobs']

fig, ax = plt.subplots(1,1)
ax.scatter(df[1,:,0], df[1,:,1], marker='H', s=230, cmap='RdBu_r', c=df[1,:,4], vmin=-0.5, vmax=0.5)
ax.quiver(df[1,:,0], df[1,:,1], df[1,:,2], df[1,:,3], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
ax.set_aspect('equal')

mx = np.asarray([np.min(df[1,:,0]),np.max(df[1,:,0])])
my = np.asarray([np.min(df[1,:,1]),np.max(df[1,:,1])])
ax.set_xlim(1.25*mx)
ax.set_ylim(1.25*my)
ax.axis('off')
plt.savefig('test.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.close()
