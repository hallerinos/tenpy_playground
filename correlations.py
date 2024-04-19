import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['figure.figsize'] = (2.0*(3.0+3.0/8.0),(3.0+3.0/8.0))
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = '/work/projects/tmqs_projects/skyrmion_liquid/Ly7'
fn = f'{dir}/dmrg_chi_128_Bz_-0.6156_Lx_14_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_128_Bz_-0.5062_Lx_14_Ly_7_bc_infinite.h5'

dir = '/work/projects/tmqs_projects/skyrmion_liquid/out'
fn = f'{dir}/dmrg_chi_64_Bz_-0.6500_Lx_3_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_64_Bz_-0.6250_Lx_11_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_128_Bz_-0.6219_Lx_13_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_256_Bz_-0.6125_Lx_14_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_256_Bz_-0.6125_Lx_8_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_256_Bz_-0.6250_Lx_11_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_256_Bz_-0.6250_Lx_6_Ly_7_bc_infinite.h5'
fn = f'{dir}/dmrg_chi_512_Bz_-0.6250_Lx_12_Ly_7_bc_infinite.h5'

dir = '/work/projects/tmqs_projects/skyrmion_liquid/Ly9_different_Lx_1'
fn = f'{dir}/dmrg_chi_128_Bz_-0.5078_Lx_30_Ly_9_bc_infinite.h5'

out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

try:
    with h5py.File(fn, 'r') as f:
        # or for partial reading:
        psi = hdf5_io.load_from_hdf5(f, "/psi")
        mms = hdf5_io.load_from_hdf5(f, "/measurements")
        sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
except:
    print(f'file: {fn} not readable')

# cL = psi.correlation_length()
# print(cL/psi.L)

for o12 in ['zz','xx', 'yy', 'xy', 'yz']:
        fig, axs = plt.subplots(3,2,sharex=True,sharey=False)
        axs = axs.ravel()

        a, b = o12
        o1 = f'S{a}'
        o2 = f'S{b}'
        print(o1, o2)
        # for jy in range(sim['model_params']['Lx']):
        for jy in [2,3,4]:
            pos = 4 + sim['model_params']['Ly']*jy
            print(jy, pos)
            sites1 = [pos]
            step = 1
            sites2 = range(pos+step, 6*psi.L+pos+1, step)

            corr = np.asarray(psi.correlation_function(o1, o2,  sites1=sites1, sites2=sites2))
            conn1 = np.asarray(psi.expectation_value(o1, sites=sites1))
            conn2 = np.asarray(psi.expectation_value(o2, sites=sites2))
            conn1 = np.reshape(conn1, (1, len(conn1)))
            conn2 = np.reshape(conn2, (1, len(conn2)))
            conn = np.matmul(np.transpose(conn1), conn2)
            ursell = corr - conn

            for (idy,Ys) in enumerate([corr, conn, ursell]):
                data_re = np.real(Ys[0])
                data_im = np.imag(Ys[0])
                im = axs[2*idy].plot(np.asarray(sites2)/psi.L, np.abs(data_re), alpha=0.4, linewidth=0.2)
                col = im[0].get_color()
                axs[2*idy].scatter(np.asarray(sites2)[::sim['model_params']['Ly']]/psi.L, np.abs(data_re[::sim['model_params']['Ly']]), alpha=0.4, color=col)
                axs[2*idy].plot(np.asarray(sites2)[::psi.L]/psi.L, np.abs(data_re[::psi.L]), color=col)
                if np.linalg.norm(data_im) < 1e-5: continue
                axs[2*idy+1].plot(np.asarray(sites2)/psi.L, data_im)
        [ax.set_yscale('log') for ax in axs]

        so1 = '\\hat S_{i,' + a + '}'
        so2 = '\\hat S_{i+n,' + b + '}'
        axs[0].set_title(f'$|\\Re\\langle {so1} {so2}\\rangle|$')
        axs[1].set_title(f'$|\\Im\\langle {so1} {so2}\\rangle|$')
        axs[2].set_title(f'$|\\Re\\langle {so1} \\rangle \\langle {so2}\\rangle|$')
        axs[3].set_title(f'$|\\Im\\langle {so1} \\rangle \\langle {so2}\\rangle|$')
        axs[4].set_title(f'$|\\Re\\left(\\langle {so1} {so2}\\rangle - \\langle {so1} \\rangle \\langle {so2}\\rangle\\right)|$')
        axs[5].set_title(f'$|\\Im\\left(\\langle {so1} {so2}\\rangle - \\langle {so1} \\rangle \\langle {so2}\\rangle\\right)|$')

        axs[-2].set_xlabel('$n/N$')
        axs[-1].set_xlabel('$n/N$')

        fn_repl = fn.replace(dir,'').replace('.h5','')
        corr_fn = f'{out_dir}/{fn_repl}_correlations_{a}{b}.png'
        fig.tight_layout()
        plt.savefig(corr_fn, dpi=600, bbox_inches='tight')
        plt.close()
        print(corr_fn)