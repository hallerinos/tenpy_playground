import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (2.0*(3.0+3.0/8.0),(3.0+3.0/8.0))
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out'
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

chis = ['8']
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]

# lxs = range(3,16)
# lxs = [4,5,6,7,8,9,10,11]
# sstr = [f'*chi_{128}*Lx_{lx}*finite.h5' for lx in lxs]
# lxs = range(3,16)
# lxs = [4,5,6,7,8,9,10,11]
# sstr = [f'*chi_{128}*Lx_{lx}*finite.h5' for lx in lxs]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

check_convergence = True
skip_bad = False
plot_snap = True
plot_mz = True
ms = 14
mkr = 's'

for (idfns,fns) in enumerate(fnss):
    av_Mz = np.zeros((2,len(fns)))
    av_norm = np.zeros((2,len(fns)))
    conv_cols = []
    for (idfn,fn) in enumerate(fns):
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                ene = hdf5_io.load_from_hdf5(f, "/energy")
                mms = hdf5_io.load_from_hdf5(f, "/measurements")
                if check_convergence: sws = hdf5_io.load_from_hdf5(f, "/sweep_stats")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except:
            print(f'file: {fn} not readable')
            continue

        if check_convergence:
            convE = sws['Delta_E'][-5:]
            convS = sws['Delta_S'][-5:]
            converged = True
            if np.max(np.abs(convE)) > 1e-6:
                print(f'{fn}')
                print(f'Bad convergence: dE={convE}')
                conv_cols.append('grey')
                converged = False
            elif np.max(np.abs(convS)) > 1e-2:
                print(f'{fn}')
                print(f'Bad convergence: dS={convS}')
                conv_cols.append('cyan')
                converged = False
            else:
                conv_cols.append('black')

        df = mms['lobs']
        if len(df)<2:
            print('Final observables not available... skipping...')
            continue
        df = df[1]

        df_re = np.real(df)
        df_im = np.imag(df)

        # this should all be zeros
        # print([np.linalg.norm(df_im[:,i]) for i in range(5)])
        # this might be nonzero...
        # print([np.linalg.norm(df_im[:,i]) for i in [5,6]])

        norm = (df_re[:,2]**2+df_re[:,3]**2+df_re[:,4]**2)**0.5

        av_Mz[:,idfn] = -1*sim['model_params']['Bz'], -np.mean(df_re[:,4])
        av_norm[:,idfn] = -1*sim['model_params']['Bz'], np.mean(norm)

        if plot_snap:
            if skip_bad:
                if check_convergence and not converged:
                    continue
            fig, axs = plt.subplots(2,1)
            axs = axs.ravel()
            ax = axs[0]
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=df_re[:,4], vmin=-0.5, vmax=0.5)
            ax.quiver(df_re[:,0], df_re[:,1], df_re[:,2], df_re[:,3], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
            ax.set_aspect('equal')
            ax.set_title('$\\langle \\vec S_{i} \\rangle$')

            fig.colorbar(imag)

            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')

            ax = axs[1]
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=norm, vmin=min(norm), vmax=max(norm))
            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')
            ax.set_aspect('equal')
            fig.colorbar(imag)
            ax.set_title('$|\\langle \\vec S_{i} \\rangle|$')

            fig.suptitle(f'Energy density {ene}', fontsize=16)

            fn_repl = fn.replace('.h5', '').replace(dir, out_dir)
            fn_fig_out = f'{fn_repl}.jpg'
            plt.tight_layout()
            plt.savefig(fn_fig_out, dpi=1200, bbox_inches='tight')
            plt.close()
            print(fn_fig_out)

            # exit()

    if not plot_mz: continue

    try:
        Bzs = np.unique(av_Mz[:,0])
        mzs = np.unique(av_Mz[:,1])
        fig_mag, ax_mag = plt.subplots(1,2)
        ax_mag = ax_mag.ravel()
        ax = ax_mag[0]
        ax.scatter(*av_Mz, color=conv_cols)
        ax.set_xlabel('$B_z/J$')
        ax.set_ylabel('$-m_z$')
        ax.set_xlim([0.5,0.8])
        ax.set_ylim([-0.45,-0.3])
        # ax.set_ylim([np.minimum(av_Mz[1,:]), np.maximum(av_Mz[1,:])])
        # ax.set_title(f'System sizes: {sim['model_params']['Lx']}$\\times${sim['model_params']['Ly']}')

        ax = ax_mag[1]
        ax.scatter(*av_norm, color=conv_cols)
        ax.set_xlabel('$B_z/J$')
        ax.set_ylabel('average spin norm')
        ax.set_xlim([0.5,0.8])
        ax.set_ylim([0.35,0.51])
        # ax.set_xlim([np.minimum(av_norm[0,:]), np.maximum(av_norm[0,:])])
        # ax.set_ylim([np.minimum(av_norm[1,:]), np.maximum(av_norm[1,:])])
        # ax.set_title(f'System sizes: {sim['model_params']['Lx']}$\\times${sim['model_params']['Ly']}')
        fn = f'{out_dir}/av_Mz_Lx_{sim['model_params']['Lx']}_Ly_{sim['model_params']['Ly']}_chi_{chis[0]}.jpg'
        plt.tight_layout()
        plt.savefig(fn, dpi=1200, bbox_inches='tight')
        plt.close()

        print(fn)
    except:
        print('Could not plot magnetization... skipping...')

