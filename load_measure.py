import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
from aux.plot_lobs import plot_lobs
import os, copy
from chiral_magnet import my_square, my_triangular
from my_correlation import concurrence

import pandas as pd

dir = 'out/'

chis = [64]
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

compute_correlations = False

out_dir = 'plots/'
os.makedirs(out_dir, exist_ok=True)

for fns in fnss:
    for fn in fns:
        print(f'Starting to measure: {fn}')
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                ene = hdf5_io.load_from_hdf5(f, "/energy")
                psi = hdf5_io.load_from_hdf5(f, "/psi")
                # mms = hdf5_io.load_from_hdf5(f, "/measurements")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except Exception as err:
            print(f'file: {fn} not readable')
            print(f'error: {err}')
            continue
        print(f'{fn} loaded')

        mps = sim['model_params']
        lat = eval(sim['model_params']['lattice'])(mps['Lx'], mps['Ly'], None, bc=[mps['bc_x'], mps['bc_y']])
        pos = np.asarray([lat.position(lat.mps2lat_idx(i)) for i in range(psi.L)])
        pos_av = np.mean(pos, axis=0)
        pos = pos - pos_av
        print('Lattice recreated.')

        # make sure the MPS it's in canonical form
        if not np.linalg.norm(psi.norm_test()) < 1e-10:
            print(f'Norm test failed. Continue with next measurement.')
            continue
        # psi.canonical_form()

        exp_Sx = psi.expectation_value("Sx")
        print('Computed Sx')
        exp_Sy = psi.expectation_value("Sy")
        print('Computed Sy')
        exp_Sz = psi.expectation_value("Sz")
        print('Computed Sz')
        # exp_Sp = psi.expectation_value("Sp")
        # print('Computed Sp')
        # exp_Sm = psi.expectation_value("Sm")
        # print('Computed Sm')

        vNEE = psi.entanglement_entropy()
        # vNEE = np.append(vNEE, 0)
        print('Computed vNEE')

        if compute_correlations:
            xs, ys = np.meshgrid(pos[:,0], pos[:,1])
            xs, ys = xs.flatten(), ys.flatten()
            cc = abs(concurrence(psi, 'Sy', 'Sy').flatten())
            print('Computed concurrence')

            df = pd.DataFrame()
            df['x'] = xs
            df['y'] = ys
            df['cc'] = cc
            fn_csv = fn.replace('.h5','_corr.csv')
            df.to_csv(fn_csv)


        abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

        df = pd.DataFrame()
        df['x'] = pos[:,0]
        df['y'] = pos[:,1]
        df['S_x'] = exp_Sx
        df['S_y'] = exp_Sy
        df['S_z'] = exp_Sz
        df['vNEE'] = vNEE
        # df['S_+'] = exp_Sp
        # df['S_-'] = exp_Sm
        df['S'] = (df['S_x']**2 + df['S_y']**2 + df['S_z']**2)**0.5
        fn_csv = fn.replace('.h5','_lobs.csv')
        df.to_csv(fn_csv)

        print(f'expectation values computed and CSV saved: {fn_csv}')

        fn_repl = fn.replace(dir, out_dir).replace('.h5', '.jpg')
        plot_lobs(df, fn_repl, mkr='h', ms=800, title=f'Energy density: {ene}')
