"""Example to extract the central charge from the entranglement scaling.

This example code evaluate the central charge of the transverse field Ising model using IDMRG.
The expected value for the central charge c = 1/2. The code always recycle the environment from
the previous simulation, which can be seen at the "age".

For the theoretical background why :math:`S = c/6 log(xi)`, see :cite:`pollmann2009`.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import tenpy
import time
from tenpy.networks.mps import MPS
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import BosonSite, set_common_charges
import tenpy.models.lattice as lattice

class TwoComponentBoseHubbardModel(CouplingMPOModel):
    def init_sites(self, model_params):
        n_max = model_params.get('n_max', 3)
        filling = model_params.get('filling', 1)
        conserve = model_params.get('conserve', 'N')
        if conserve == 'best':
            conserve = 'N'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        siteA = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        siteB = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        set_common_charges([siteA, siteB], 'independent')
        return [siteA, siteB]

    def init_lattice(self, model_params):
        pairs = {'onsite': [(0, 1, np.array([0]))], 'nearest_neighbors': [(0, 0, np.array([1])), (1, 1, np.array([1]))]}
        L = model_params.get('L', 10)
        sites = self.init_sites(model_params)
        lat = lattice.Lattice([L], sites, pairs = pairs)
        return lat	

    def init_terms(self, model_params):
        t = model_params.get('t', 1.)
        U = model_params.get('U', 10.)
        UAB = model_params.get('UAB', 0.)
        mu = model_params.get('mu', 0)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu - U / 2., u, 'N')
            self.add_onsite(U / 2., u, 'NN')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Bd', u2, 'B', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['onsite']:
            self.add_coupling(UAB, u1, 'N', u2, 'N', dx)

model_params=dict(L=10, n_max=4, t=-1, U=0, UAB=0, mu=0, conserve=None)
M=TwoComponentBoseHubbardModel(model_params) 
psi=MPS.from_lat_product_state(M.lat,[[1,1]])
dmrg_params = { 
        'mixer': True,  
        # 'max_E_err': 1.e-10, 
        'trunc_params': {
            'chi_max': 4,
            'svd_min': 1.e-10  
        },
        'combine': True 
    }
eng_dmrg = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi= eng_dmrg.run()