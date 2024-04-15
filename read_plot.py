import h5py
from tenpy.tools import hdf5_io
import numpy as np
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3.0+3.0/8.0),(3.0+3.0/8.0))
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out'
chis = ['64', '128', '256', '512']
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

plot_all = False

maxlen = np.max([len(f) for f in fnss])
av_Mz = np.zeros((len(fnss),2,maxlen))
for (idfns,fns) in enumerate(fnss):
    for (idfn,fn) in enumerate(fns):
        print(fn)
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                mms = hdf5_io.load_from_hdf5(f, "/measurements")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except:
            print(f'file: {fn} not readable')
            continue

        # print(sim['model_params'])
        df = mms['lobs']
        if len(df)<2: continue
        df = df[1]

        df_re = np.real(df)
        df_im = np.imag(df)

        # this should all be zeros
        # print([np.linalg.norm(df_im[:,i]) for i in range(5)])
        # this might be nonzero...
        # print([np.linalg.norm(df_im[:,i]) for i in [5,6]])

        av_Mz[idfns,:,idfn] = -1*sim['model_params']['Bz'], np.mean(df_re[:,4])

        if plot_all:
            ms = 30
            mkr = 'H'

            fig, axs = plt.subplots(2,2)
            axs = axs.ravel()
            ax = axs[0]
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=df_re[:,4], vmin=-0.5, vmax=0.5)
            ax.quiver(df_re[:,0], df_re[:,1], df_re[:,2], df_re[:,3], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
            ax.set_aspect('equal')

            fig.colorbar(imag)

            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')

            norm = (df_re[:,2]**2+df_re[:,3]**2+df_re[:,4]**2)**0.5

            ax = axs[1]
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=norm, vmin=min(norm), vmax=max(norm))
            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')
            ax.set_aspect('equal')
            fig.colorbar(imag)

            ax = axs[2]
            spabs = (df_re[:,5]**2+df_im[:,5]**2)**0.5
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=spabs)
            ax.quiver(df_re[:,0], df_re[:,1], df_re[:,5], df_im[:,5], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')
            ax.set_aspect('equal')
            fig.colorbar(imag)

            ax = axs[3]
            smabs = (df_re[:,6]**2+df_im[:,6]**2)**0.5
            imag = ax.scatter(df_re[:,0], df_re[:,1], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=smabs)
            ax.quiver(df_re[:,0], df_re[:,1], df_re[:,6], df_im[:,6], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
            mx = np.asarray([np.min(df_re[:,0]),np.max(df_re[:,0])])
            my = np.asarray([np.min(df_re[:,1]),np.max(df_re[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')
            ax.set_aspect('equal')
            fig.colorbar(imag)

            fn_repl = fn.replace('.h5', '')
            plt.savefig(f'{fn_repl}.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
            plt.close()

Bzs = np.unique(av_Mz[:,0,:])
mzs = np.unique(av_Mz[:,1,:])
fig_mag, ax_mag = plt.subplots(1,1)
lbls = [f'$\\chi = {s}$' for s in chis]
[ax_mag.scatter(*av_Mz[i], label=lbls[i]) for i in range(len(fnss))]
ax_mag.set_xlabel('$B_z/J$')
ax_mag.set_ylabel('$m_z$')
ax_mag.set_xlim(Bzs[[1,-1]])
ax_mag.set_ylim(mzs[[1,-1]])
plt.legend()
plt.savefig(f'{dir}/av_Mz.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.close

