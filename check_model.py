import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd, vumps
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from chiral_magnet import *

# compute local observables
def compute_lobs(psi):
    exp_Sx = psi.expectation_value("Sx")
    exp_Sy = psi.expectation_value("Sy")
    exp_Sz = psi.expectation_value("Sz")

    abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

    return [exp_Sx, exp_Sy, exp_Sz, abs_exp_Svec]

bond_dim = 32

model_params = {
        'bc_x': 'periodic', 'bc_y': 'periodic', 'bc_MPS': 'finite',
        'Lx' : 5, 'Ly': 5, 'lattice': 'my_square', 'conserve': None,
    }

M = chiral_magnet_square(model_params)

sites = M.lat.mps_sites()

psi = MPS.from_desired_bond_dimension(sites, bond_dim, bc='finite')

# generate a random initial state
TEBD_params = {'N_steps': 1, 'trunc_params':{'chi_max': bond_dim}}
eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
eng.run()

psi.canonical_form()
lobs = compute_lobs(psi)

dmrg_params = {
    'mixer': 'SubspaceExpansion',  # no subspace expansion
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 2.0,
        'disable_after': 10,
    },
    'diag_method': 'lanczos',
    'lanczos_params': {
        'N_max': 5,  # fix the number of Lanczos iterations: the number of `matvec` calls
        'N_min': 2,
        'N_cache': 5,  # keep the states during Lanczos in memory
        'reortho': False,
    },
    'N_sweeps_check': 1,
    'max_E_err': 1e-8,
    'max_S_err': 1e-6,
    'max_sweeps': 10,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    },
}
vumps_params = {
    'mixer': None,  # no subspace expansion
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 2.0,
        'disable_after': 10,
    },
    'diag_method': 'lanczos',
    'lanczos_params': {
        'N_max': 5,  # fix the number of Lanczos iterations: the number of `matvec` calls
        'N_min': 2,
        'N_cache': 5,  # keep the states during Lanczos in memory
        'reortho': False,
    },
    'N_sweeps_check': 10,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-14,
    },
    'max_sweeps': 10,
    'max_split_err': 1e-8,  # different criteria than DMRG
    'max_E_err': 1e-8,
    'max_S_err': 1.e-8,
}

# eng = dmrg_parallel.DMRGThreadPlusHC(psi, M, dmrg_params)
eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
# eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
# eng = vumps.SingleSiteVUMPSEngine(psi, M, vumps_params)

E, psi = eng.run()


# plt.figure(figsize=(5, 6))
# ax = plt.gca()
# lat = M.lat
# lat.plot_coupling(ax)
# lat.plot_order(ax, linestyle=':')
# lat.plot_sites(ax)
# lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
# ax.set_aspect('equal')
# ax.set_xlim(-1)
# ax.set_ylim(-1)
# plt.savefig('lattice_2.png')