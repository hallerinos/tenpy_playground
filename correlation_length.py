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

dir = '/work/projects/tmqs_projects/skyrmion_liquid/out'
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

chis = ['128']
sstr = [f'*chi_{chi}*finite.h5' for chi in chis]

lxs = range(3,16)
lxs = [4]
sstr = [f'*chi_{64}*Lx_{lx}*finite.h5' for lx in lxs]
fnss = [np.sort(find_files(s, dir)) for s in sstr]

out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv('corr_length.csv')

df = df[df['\\xi'] < 1]

print(df.keys())

fig, ax = plt.subplots()
for M in np.unique(df['M']):
    tp = df[df['M']==M]
    Xs = np.log(np.asarray(range(1,len(tp)+1)))
    Ys = np.log(tp["\\xi"])*7*4
    ax.scatter(Xs, Ys)

    coef = np.polyfit(Xs,Ys,1)
    poly1d_fn = np.poly1d(coef)
    print(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y

    plt.plot(Xs, poly1d_fn(Xs), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

plt.tight_layout()
plt.savefig('test.jpg', dpi=600, bbox_inches='tight')

exit()

for (idfns,fns) in enumerate(fnss):
    for (idfn,fn) in enumerate(fns):
        print(fn)
        try:
            with h5py.File(fn, 'r') as f:
                # or for partial reading:
                psi = hdf5_io.load_from_hdf5(f, "/psi")
                mms = hdf5_io.load_from_hdf5(f, "/measurements")
                sim = hdf5_io.load_from_hdf5(f, "/simulation_parameters")
        except:
            print(f'file: {fn} not readable')

        cL = psi.correlation_length()
        print(cL/psi.L)