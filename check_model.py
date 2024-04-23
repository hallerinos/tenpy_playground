import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd, vumps
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

import pandas as pd

import logging.config

# create the config file for logs
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_stdout': {'class': 'logging.StreamHandler',
                 'formatter': 'custom',
                 'level': 'INFO',
                 'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout'], 'level': 'DEBUG'},
}

from chiral_magnet import *

# start logging
logging.config.dictConfig(conf)

# compute local observables
def compute_lobs(psi):
    exp_Sx = psi.expectation_value("Sx")
    exp_Sy = psi.expectation_value("Sy")
    exp_Sz = psi.expectation_value("Sz")

    abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

    return [exp_Sx, exp_Sy, exp_Sz, abs_exp_Svec]

bond_dim = 128

model_params = {
        'bc_x': 'open', 'bc_y': 'periodic', 'bc_MPS': 'finite',
        'bc_classical': True,
        'Lx' : 3, 'Ly': 4, 'lattice': 'my_square', 'conserve': None,
        'Bz' : -1.0
    }

M = chiral_magnet(model_params)

sites = M.lat.mps_sites()

psi = MPS.from_desired_bond_dimension(sites, bond_dim, bc='finite')

# generate a random initial state
# TEBD_params = {'N_steps': 1, 'trunc_params':{'chi_max': bond_dim}}
# eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
# eng.run()

# psi.canonical_form()
# lobs = compute_lobs(psi)

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
    'max_sweeps': 1,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    },
}

# eng = dmrg_parallel.DMRGThreadPlusHC(psi, M, dmrg_params)
# eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
# eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
# eng = vumps.SingleSiteVUMPSEngine(psi, M, vumps_params)

info = dmrg.run(psi, M, dmrg_params)
# print(info)
# print(info['E'])
keys = ['sweep', 'E', 'Delta_E', 'S', 'max_S', 'max_E_trunc']
df = pd.DataFrame()
for k in keys:
    df[k] = info['sweep_statistics'][k]


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