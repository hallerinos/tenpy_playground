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

dir = 'L7'
chis = ['32', '64', '128']
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

plot_all = True

print(len(fnss))
av_Mz = np.zeros((len(fnss),2,len(fnss[0])))
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

        av_Mz[idfns,:,idfn] = -1*sim['model_params']['Bz'], np.mean(df[:,4])

        if plot_all:
            ms = 70
            mkr = 'H'

            fig, axs = plt.subplots(2,1)
            axs = axs.ravel()
            ax = axs[0]
            imag = ax.scatter(df[:,0], df[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=df[:,4], vmin=-0.5, vmax=0.5)
            ax.quiver(df[:,0], df[:,1], df[:,2], df[:,3], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
            ax.set_aspect('equal')

            fig.colorbar(imag)

            mx = np.asarray([np.min(df[:,0]),np.max(df[:,0])])
            my = np.asarray([np.min(df[:,1]),np.max(df[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')

            norm = (df[:,2]**2+df[:,3]**2+df[:,4]**2)**0.5

            ax = axs[1]
            imag = ax.scatter(df[:,0], df[:,1], marker=mkr, edgecolor='None', s=ms, cmap='RdBu_r', c=norm, vmin=min(norm), vmax=max(norm))
            ax.set_aspect('equal')

            mx = np.asarray([np.min(df[:,0]),np.max(df[:,0])])
            my = np.asarray([np.min(df[:,1]),np.max(df[:,1])])
            ax.set_xlim(1.25*mx)
            ax.set_ylim(1.25*my)
            ax.axis('off')
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

