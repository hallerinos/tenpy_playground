import h5py
from tenpy.tools import hdf5_io
import numpy as np
import tenpy
from tenpy.models import lattice
from aux.find_files import find_files
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from DMI_model import *
import yaml
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (20,20)
# plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

import pandas as pd

dir = 'out/'
fn = f'{dir}dmrg_chi_512_Bz_-0.7700_Lx_251_Ly_9_bc_finite.h5'
fn_sim = 'cfg/0863.yml'
# dir = 'L7/'
# fn = f'{dir}dmrg_chi_16_Lx_7_Ly_7_Bz_-0.50_bc_infinite.h5'
# fn_sim = 'cfg/0863.yml'

with open(fn_sim, 'r') as f:
    sim = yaml.safe_load(f)
tenpy.simulations.simulation.resume_from_checkpoint(filename=fn, update_sim_params=sim)