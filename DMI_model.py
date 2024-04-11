"""This file ``/examples/DMI_model.py`` illustrates customization code for simulations.

Put this file somewhere where it can be importet by python, i.e. either in the working directory
or somewhere in the ``PYTHON_PATH``. Then you can use it with the parameters file
``/examples/DMI_model.yml`` from the terminal like this::

    tenpy-run -i DMI_model simulation_custom.yml
"""

"""Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
import tenpy.models.lattice as lat
from tenpy.tools.params import asConfig

__all__ = ['DMI_model', 'MySpinChain']


class DMI_model(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions with DMI.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`DMI_model` below.

    Options
    -------
    .. cfg:config :: DMI_model
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
        ctr = 0
        Ly = model_params['Ly']
        fig, ax = plt.subplots()
        for u1, u2, dx in nn_pairs:
            for (Ji, Si) in zip(J, Svec):
                self.add_coupling(Ji, u1, Si, u2, Si, dx)
            mps_i, mps_j, _, _ = self.lat.possible_couplings(u1, u2, dx)
            [ax.scatter(self.lat.position(self.lat.mps2lat_idx(i))[0], self.lat.position(self.lat.mps2lat_idx(i))[1], marker='H', s=400, c='black', zorder=-999) for i in mps_i]
            for i, j in zip(mps_i, mps_j):
                if i > j: # ensure proper ordering for TenPy (operators commute)
                    i, j = j, i
                ri = self.lat.position(self.lat.mps2lat_idx(i))
                rj = self.lat.position(self.lat.mps2lat_idx(j))
                dist = rj-ri
                fac = 1.0
                # this works only for Triangular chain!!!
                if np.linalg.norm(dist) > 1.001:
                    ctr += 1
                    # if dist[0] == 0: continue
                    # ax.scatter(ri[0], ri[1])
                    # ax.scatter(rj[0], rj[1])
                    # ax.scatter(dist[0], dist[1], color='black')
                    # if dist[1] > 1.0:
                    if dist[0] == 0: dist[1] = -1
                    #     dist[1] = -np.mod(dist[1],5)
                    if dist[0] != 0: dist[1] = -0.5
                        # dist[0] = np.mod(dist[0],5.*np.sqrt(3))
                        # dist[1] = -np.mod(dist[1],5)
                    # print(f'pbc term, sites: {i} : {j}')
                    # print(dist)
                    # print(ctr)
                    # fac = 2.0
                ax.quiver(ri[0], ri[1], dist[0], dist[1], units='xy', scale=1, color='red')
                Dvec = fac*D*np.cross([0,0,1], dist/np.linalg.norm(dist))
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            if abs(Dvec[k]*self.epsilon(k,l,m)) > 0 and np.linalg.norm(dist) >= 0.9:
                                self.add_coupling_term(Dvec[k]*self.epsilon(k,l,m), i, j, Svec[l], Svec[m])
        ax.set_aspect('equal')
        # ax.set_ylim([0,20])
        # ax.set_xlim([0,20])
        plt.savefig('lattice.png', bbox_inches='tight', pad_inches=0, dpi=600)
        # done


class MySpinChain(DMI_model, NearestNeighborModel):
    """The :class:`DMI_model` on a Chain, suitable for TEBD.

    See the :class:`DMI_model` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

def measure_lobs(results, psi, model, simulation, tol=0.01):
    pos = np.asarray([model.lat.position(model.lat.mps2lat_idx(i)) for i in range(psi.L)])
    pos_av = np.mean(pos, axis=0)
    pos = pos - pos_av

    exp_Sx = psi.expectation_value("Sx")
    exp_Sy = psi.expectation_value("Sy")
    exp_Sz = psi.expectation_value("Sz")

    abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))

    df = pd.DataFrame()
    df['x'] = pos[:,0]
    df['y'] = pos[:,1]
    df['S_x'] = exp_Sx
    df['S_y'] = exp_Sy
    df['S_z'] = exp_Sz

    results['lobs'] = df