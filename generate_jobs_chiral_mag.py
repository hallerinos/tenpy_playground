import pandas as pd
import numpy as np
import yaml
import tenpy
import os
import shutil

dir_out = 'cfg'

# clean directory
try:
    shutil.rmtree(dir_out)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

# create directory for the simulation yml files
os.makedirs(dir_out, exist_ok=1)

with open('chiral_magnet.yml', 'r') as file:
    default = yaml.safe_load(file)

cfg = 0

Lxs = range(6,11)

for chi_nmax in [6,7,8]:
    for ll in Lxs:
        chi_max = 2**(chi_nmax-1)
        chi_list = {i*10:2**i for i in range(0,chi_nmax)}
        default['algorithm_params']['chi_list'] = chi_list
        default['algorithm_params']['trunc_params']['chi_max'] = chi_max

        default['model_params']['Lx'] = ll
        default['model_params']['Ly'] = 5

        Bzmin = -0.6
        Bzmax = -0.4
        nBz = 32
        dBz = (Bzmax-Bzmin)/nBz
        Bzs = [np.round(Bzmin + i*dBz, decimals=4) for i in range(0, nBz)]

        for Bz in Bzs:
            default['model_params']['Bz'] = float(Bz)
            with open(f'{dir_out}/{str(cfg).zfill(4)}.yml', 'w') as outfile:
                yaml.dump(default, outfile, default_flow_style=False)
            cfg += 1
print(f'{cfg} configs created')