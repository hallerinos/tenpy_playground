import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd, vumps
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from DMI_model import DMI_model

import pandas as pd

import os

import logging.config

# compute local observables
def compute_lobs(psi):
    exp_Sx = psi.expectation_value("Sx")
    exp_Sy = psi.expectation_value("Sy")
    exp_Sz = psi.expectation_value("Sz")

    abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

    return [exp_Sx, exp_Sy, exp_Sz, abs_exp_Svec]

# save and plot local observables
def save_plot_lobs(lobs, M, psi, fn=''):
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

    df.to_csv(f'{fn}.csv')

    fig, ax = plt.subplots(1,1)
    ax.scatter(pos[:,0], pos[:,1], marker=mkr, s=sze, cmap='RdBu_r', c=df['S_z'], vmin=-0.5, vmax=0.5)
    ax.quiver(pos[:,0], pos[:,1], df['S_x'], df['S_y'], units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
    ax.set_aspect('equal')

    mx = np.asarray([np.min(pos[:,0]),np.max(pos[:,0])])
    my = np.asarray([np.min(pos[:,1]),np.max(pos[:,1])])
    ax.set_xlim(1.25*mx)
    ax.set_ylim(1.25*my)
    ax.axis('off')
    plt.savefig(f'{fn}.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.close()

Bzs = np.linspace(-0.0, -0.3, 11)
Bzs = [-0.7, -0.6, -0.5, -0.4]
Bzs = [-1]

for Bz in Bzs:
    Bx = By = 0.0
    D = 1.0
    Jx = Jy = Jz = -0.5
    dir_out = f'results/B_{-Bz}'

    # create the config file for logs
    conf = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
        'handlers': {'to_file_info': {'class': 'logging.FileHandler',
                                'filename': f'{dir_out}/info.log',
                                'formatter': 'custom',
                                'level': 'INFO',
                                'mode': 'a'},
                    'to_file_warn': {'class': 'logging.FileHandler',
                                'filename': f'{dir_out}/warn.log',
                                'formatter': 'custom',
                                'level': 'WARN',
                                'mode': 'a'},
                    'to_stdout': {'class': 'logging.StreamHandler',
                                'formatter': 'custom',
                                'level': 'INFO',
                                'stream': 'ext://sys.stdout'}},
        'root': {'handlers': ['to_stdout', 'to_file_info', 'to_file_warn'], 'level': 'DEBUG'},
    }

    # make output directory
    os.makedirs(dir_out, exist_ok=True)
    # start logging
    logging.config.dictConfig(conf)

    bc_MPS, N_sweeps, E_tol, S_tol, bond_dim = 'finite', 1000, 1e-8, 1e-8, 10
    lattice, length, mkr, sze, bc_lat_y = 'Square', 9, 's', 500, 'open'

    # speedup for small chi
    tenpy.tools.optimization.optimize(3)

    model_params = {
        'J': [Jx, Jy, Jz],
        'B': [Bx, By, Bz],
        'D' : D,
        'bc_y': bc_lat_y, 'bc_MPS': bc_MPS,
        'Lx' : length, 'Ly': length, 'lattice': lattice, 'conserve': None,
    }

    M = DMI_model(model_params)

    sites = M.lat.mps_sites()

    psi = MPS.from_desired_bond_dimension(sites, bond_dim, bc=bc_MPS)

    # p_state = ['down']*len(sites)
    # psi = MPS.from_product_state(sites, p_state, bc=bc_MPS)

    # generate a random initial state
    TEBD_params = {'N_steps': 1, 'trunc_params':{'chi_max': bond_dim}}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()

    psi.canonical_form()
    lobs = compute_lobs(psi)
    save_plot_lobs(lobs, M, psi, fn=f'{dir_out}/magnetization_i')

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
        'max_E_err': E_tol,
        'max_S_err': S_tol,
        'max_sweeps': N_sweeps,
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
        'max_sweeps': N_sweeps,
        'max_split_err': 1e-8,  # different criteria than DMRG
        'max_E_err': E_tol,
        'max_S_err': 1.e-8,
    }

    # eng = dmrg_parallel.DMRGThreadPlusHC(psi, M, dmrg_params)
    eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
    # eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    # eng = vumps.SingleSiteVUMPSEngine(psi, M, vumps_params)

    E, psi = eng.run()
    psi.canonical_form()
    lobs = compute_lobs(psi)
    save_plot_lobs(lobs, M, psi, fn=f'{dir_out}/magnetization_f')

    # print("corr. length =", psi.correlation_length())

    data = {"psi": psi,
            "model": M,
            "parameters": model_params}

    with h5py.File(f"{dir_out}/save.h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
