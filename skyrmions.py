import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd, vumps
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from DMI_model import MySpinModel

import pandas as pd

tenpy.tools.misc.setup_logging(to_stdout="INFO")

def compute_lobs(psi):
    exp_Sx = psi.expectation_value("Sx")
    exp_Sy = psi.expectation_value("Sy")
    exp_Sz = psi.expectation_value("Sz")

    abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

    return [exp_Sx, exp_Sy, exp_Sz, abs_exp_Svec]

def save_plot_lobs(lobs, M, psi):
    pos = np.asarray([M.lat.position(M.lat.mps2lat_idx(i)) for i in range(psi.L)])
    pos_av = np.mean(pos, axis=0)
    pos = pos - pos_av

    vmin = np.min(lobs[3])
    vmax = np.max(lobs[3])

    df = pd.DataFrame()
    df['x'] = pos[:,0]
    df['y'] = pos[:,1]
    df['S_x'] = lobs[0]
    df['S_y'] = lobs[1]
    df['S_z'] = lobs[2]

    df.to_csv('lobs.csv')

    fig, ax = plt.subplots(1,1)
    ax.scatter(pos[:,0], pos[:,1], marker=mkr, s=sze, cmap='RdBu_r', c=df['S_z'], vmin=-0.5, vmax=0.5)
    ax.quiver(pos[:,0], pos[:,1], df['S_x'], df['S_y'], units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
    ax.set_aspect('equal')

    mx = np.asarray([np.min(pos[:,0]),np.max(pos[:,0])])
    my = np.asarray([np.min(pos[:,1]),np.max(pos[:,1])])
    ax.set_xlim(1.25*mx)
    ax.set_ylim(1.25*my)
    ax.axis('off')
    plt.savefig("snap.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


Bx = By = 0.0
Bz = -2
D = 5.0
Jx = Jy = Jz = -0.5

bc_MPS, N_sweeps, E_tol, bond_dim = 'infinite', 1000, 1e-8, 32
lattice, mkr, sze, L = 'my_triangular', 'h', 800, 7
bc_lat_y = 'open'

model_params = {
    'J': [Jx, Jy, Jz],
    'B': [Bx, By, Bz],
    'D' : D,
    'bc_y': bc_lat_y, 'bc_MPS': bc_MPS,
    'Lx' : L, 'Ly': L, 'lattice': lattice, 'conserve': None
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
lobs = compute_lobs(psi)
save_plot_lobs(lobs, M, psi)

dmrg_params = {
    'mixer': True,  # no subspace expansion
    # 'diag_method': 'lanczos',
    # 'lanczos_params': {
    #     # https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.lanczos.LanczosGroundState.html#cfg-config-Lanczos
    #     'N_max': 2,  # fix the number of Lanczos iterations: the number of `matvec` calls
    #     'N_min': 2,
    #     'N_cache': 10,  # keep the states during Lanczos in memory
    #     'reortho': False,
    # },
    'N_sweeps_check': 4,
    'max_E_err': E_tol,
    'max_sweeps': N_sweeps,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    }
}
vumps_params = {
    'combine': False,
    'mixer': False,
    'N_sweeps_check': 4,
    'trunc_params': {
        'chi_max': 32,
        'svd_min': 1.e-14,
    },
    'min_sweeps': 2,
    'max_sweeps': 4,
    'max_split_err': 1e-8,  # different criteria than DMRG
    'max_E_err': 1.e-12,
    'max_S_err': 1.e-8,
}

eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params) 
# eng = vumps.SingleSiteVUMPSEngine(psi, M, vumps_params) 
E, psi = eng.run()
psi.canonical_form()
lobs = compute_lobs(psi)
save_plot_lobs(lobs, M, psi)

print("corr. length =", psi.correlation_length())

data = {"psi": psi,
        "model": M,
        "parameters": model_params}

with h5py.File("save.h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)
