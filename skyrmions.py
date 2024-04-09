import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from DMI_model import MySpinModel

import pandas as pd

tenpy.tools.misc.setup_logging(to_stdout="INFO")

Bx = By = 0.0
Bz = -1.0
D = 1.0
Jx = Jy = Jz = -0.5

bc_MPS, N_sweeps, E_tol, bond_dim = 'finite', 1000, 1e-6, 32
# lattice, mkr, sze, L = 'Square', 's', 9, 9
lattice, mkr, sze, L = 'Triangular', 'H', 500, 7
bc_lat_y = 'open'

model_params = {
    'J': [Jx, Jy, Jz],
    'B': [Bx, By, Bz],
    'D' : D,
    'bc_y': bc_lat_y, 'bc_MPS': bc_MPS,
    'Lx' : 2*L, 'Ly': L, 'lattice': lattice, 'conserve': None
}

M = MySpinModel(model_params)

sites = M.lat.mps_sites()
p_state = ['down']*len(sites)
psi = MPS.from_product_state(sites, p_state, bc=bc_MPS)

# generate a random initial state
TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': bond_dim}}
eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
eng.run()
psi.canonical_form()

dmrg_params = {
    'mixer': None,  # no subspace expansion
    'diag_method': 'lanczos',
    'lanczos_params': {
        # https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.lanczos.LanczosGroundState.html#cfg-config-Lanczos
        'N_max': 3,  # fix the number of Lanczos iterations: the number of `matvec` calls
        'N_min': 3,
        'N_cache': 20,  # keep the states during Lanczos in memory
        'reortho': False,
    },
    'max_E_err': E_tol,
    'max_sweeps': N_sweeps,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    }
}
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params) 
E, psi = eng.run()
psi.canonical_form()

exp_Sx = psi.expectation_value("Sx")
exp_Sy = psi.expectation_value("Sy")
exp_Sz = psi.expectation_value("Sz")

abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))
vmin = np.min(abs_exp_Svec)
vmax = np.max(abs_exp_Svec)

pos = np.asarray([M.lat.position(M.lat.mps2lat_idx(i)) for i in range(psi.L)])
pos_av = np.mean(pos)
pos = pos - pos_av

df = pd.DataFrame()
df['x'] = pos[:,0]
df['y'] = pos[:,1]
df['Sx'] = exp_Sx
df['Sy'] = exp_Sy
df['Sz'] = exp_Sz

df.to_csv('lobs.csv')

fig, ax = plt.subplots(1,1)
ax.scatter(pos[:,0], pos[:,1], marker=mkr, s=sze, cmap='RdBu_r', c=exp_Sz, vmin=-0.5, vmax=0.5)
ax.quiver(pos[:,0], pos[:,1], exp_Sx, exp_Sy, units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
ax.set_aspect('equal')

mmx = np.asarray([np.min(pos[:,0]),np.max(pos[:,0])])
mmy = np.asarray([np.min(pos[:,1]),np.max(pos[:,1])])
ax.set_xlim(1.25*mmx)
ax.set_ylim(1.25*mmy)
ax.axis('off')
plt.tight_layout()
plt.savefig("snap.jpg", dpi=300)
plt.close()

data = {"psi": psi,
        "model": M,
        "parameters": model_params}

with h5py.File("save.h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)
