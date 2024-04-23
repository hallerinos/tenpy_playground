import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd, vumps
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

import pandas as pd

import yaml

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

with open('cfg/0001.yml', 'r') as file:
    params = yaml.safe_load(file)

print(params['model_params'])

M = chiral_magnet(params['model_params'])

sites = M.lat.mps_sites()

psi = MPS.from_desired_bond_dimension(sites, 1, bc=params['model_params']['bc_MPS'])

info = dmrg.run(psi, M, params['algorithm_params'])
keys = ['sweep', 'E', 'Delta_E', 'S', 'max_S', 'max_E_trunc']
df = pd.DataFrame()
for k in keys:
    df[k] = info['sweep_statistics'][k]