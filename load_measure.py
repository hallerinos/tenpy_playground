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
fn = f'{dir}dmrg_chi_64_Bz_-0.6125_Lx_251_Ly_9_bc_finite.h5'

chis = [128]
sstr = [f'*chi_{chi}*Lx_251*finite.h5' for chi in chis]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

out_dir = 'plots/'
os.makedirs(out_dir, exist_ok=True)

for fns in fnss:
    for fn in fns:
        print(f'Starting to measure: {fn}')
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                psi = hdf5_io.load_from_hdf5(f, "/psi")
                mms = hdf5_io.load_from_hdf5(f, "/measurements")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except:
            print(f'file: {fn} not readable')
        print(f'{fn} loaded')

        M = DMI_model(sim['model_params'])
        pos = np.asarray([M.lat.position(M.lat.mps2lat_idx(i)) for i in range(psi.L)])
        pos_av = np.mean(pos, axis=0)
        pos = pos - pos_av

        exp_Sx = psi.expectation_value("Sx")
        print('Sx computed')
        exp_Sy = psi.expectation_value("Sy")
        print('Sy computed')
        exp_Sz = psi.expectation_value("Sz")
        print('Sz computed')
        exp_Sp = psi.expectation_value("Sp")
        print('Sp computed')
        exp_Sm = psi.expectation_value("Sm")
        print('Sm computed')

        abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

        df = pd.DataFrame()
        df['x'] = pos[:,0]
        df['y'] = pos[:,1]
        df['S_x'] = exp_Sx
        df['S_y'] = exp_Sy
        df['S_z'] = exp_Sz
        df['S_+'] = exp_Sp
        df['S_-'] = exp_Sm
        df['S'] = (df['S_x']**2 + df['S_y']**2 + df['S_z']**2)**0.5

        df.to_csv(fn.replace('.h5','.csv'))

        print('expectation values computed and CSV saved')

        mkr = 'H'
        ms = 30

        fig, ax = plt.subplots(1,1)
        imag = ax.scatter(df['x'], df['y'], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=df['S'], vmin=min(df['S']), vmax=max(df['S']))
        ax.quiver(df['x'], df['y'], df['S_x'], df['S_y'], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
        ax.set_aspect('equal')
        ax.set_title('$\\langle \\vec S_{i} \\rangle$')

        fig.colorbar(imag)
        ax.axis('off')

        fn_repl = fn.replace(dir, out_dir).replace('.h5', '.jpg')
        plt.tight_layout()
        plt.savefig(fn_repl, dpi=300, bbox_inches='tight')
        plt.close()
        print(fn_repl)