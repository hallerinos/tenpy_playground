"""Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
import tenpy.models.lattice as lat
from tenpy.tools.params import asConfig

__all__ = ['MySpinModel', 'MySpinChain']


class MySpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions with DMI.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`MySpinModel` below.

    Options
    -------
    .. cfg:config :: MySpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', None)
        site = SpinSite(S, conserve)
        return site

    def epsilon(self,i,j,k):
        if [i,j,k] in [[0,1,2], [1,2,0], [2,0,1]]:
            return +1
        elif [i,j,k] in [[1,0,2], [2,1,0], [0,2,1]]:
            return -1
        else:
            return 0

    def init_terms(self, model_params):
        J = model_params.get('J', [1.,1.,1.])
        B = model_params.get('B', [0.,0.,0.])
        D = model_params.get('D', 0.)

        Svec = ['Sx', 'Sy', 'Sz']

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            for (Bi, Si) in zip(B, Svec):
                self.add_onsite(Bi, u, Si)
        
        nn_pairs = self.lat.pairs['nearest_neighbors']
        for u1, u2, dx in nn_pairs:
            for (Ji, Si) in zip(J, Svec):
                self.add_coupling(Ji, u1, Si, u2, Si, dx)
            mps_i, mps_j, _, _ = self.lat.possible_couplings(u1, u2, dx)
            for i, j in zip(mps_i, mps_j):
                if i > j: # ensure proper ordering for TenPy (operators commute)
                    i, j = j, i
                ri = self.lat.position(self.lat.mps2lat_idx(i))
                rj = self.lat.position(self.lat.mps2lat_idx(j))
                dist = rj-ri
                if np.linalg.norm(dist) > 1.1:
                    # print('pbc term')
                    # print(dist)
                    dist *= -1
                Dvec = D*np.cross([0,0,1], dist/np.linalg.norm(dist))
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            if abs(Dvec[k]*self.epsilon(k,l,m)) > 0 and np.linalg.norm(dist) >= 0.9:
                                self.add_coupling_term(Dvec[k]*self.epsilon(k,l,m), i, j, Svec[l], Svec[m])
        # done


class MySpinChain(MySpinModel, NearestNeighborModel):
    """The :class:`MySpinModel` on a Chain, suitable for TEBD.

    See the :class:`MySpinModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
