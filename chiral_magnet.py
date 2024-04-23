"""This file ``/examples/chiral_magnet.py`` illustrates customization code for simulations.

Put this file somewhere where it can be importet by python, i.e. either in the working directory
or somewhere in the ``PYTHON_PATH``. Then you can use it with the parameters file
``/examples/chiral_magnet.yml`` from the terminal like this::

    tenpy-run -i chiral_magnet simulation_custom.yml
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
from tenpy.models.lattice import Chain, SimpleLattice
import tenpy.models.lattice as lat
from tenpy.tools.params import asConfig

__all__ = ['chiral_magnet', 'MySpinChain']


class chiral_magnet(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions with DMI.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`chiral_magnet` below.

    Options
    -------
    .. cfg:config :: chiral_magnet
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
        J = np.asarray(model_params.get('J', [-0.5,-0.5,-0.5]))
        B = np.asarray(model_params.get('B', [0.,0.,0.]))
        Bz = model_params.get('Bz', -0.125)
        D = np.asarray(model_params.get('D', [0., 0., 1.0]))

        if Bz != 0: B=[0,0,Bz]

        # J = 0.*J

        Svec = ['Sx', 'Sy', 'Sz']

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            for (Bi, Si) in zip(B, Svec):
                self.add_onsite(Bi, u, Si)

        nn_pairs = self.lat.pairs['nearest_neighbors']
        # print(nn_pairs)
        ctr = 0

        fig = plt.figure()
        ax = fig.gca()
        for u1, u2, dx in nn_pairs:
            mps_i, mps_j, _, _ = self.lat.possible_couplings(u1, u2, dx)
            for i, j in zip(mps_i, mps_j):
                # print(f'Order: {i,j}')
                if i > j: # ensure proper ordering for TenPy (operators commute)
                    i, j = j, i
                    # print(f'Reorder: {i,j}')
                ri = self.lat.position(self.lat.mps2lat_idx(i))
                rj = self.lat.position(self.lat.mps2lat_idx(j))
                dist = rj-ri

                pt = False
                if np.linalg.norm(dist) > 1+1e-6:
                    if dist[0]>0:
                        if model_params['lattice'] == 'my_square':
                            dist = np.asarray([-1., 0.])
                    if dist[1]>0:
                        if model_params['lattice'] == 'my_square':
                            dist = np.asarray([0., -1.])
                    pt = True
                ax.quiver(*ri, dist[0], dist[1], units='xy', scale=1, zorder=-1)
                sc = ax.scatter(*ri, marker='x', s=100)
                col = sc.get_facecolors()[0].tolist()
                sc = ax.scatter(*(ri+dist), marker='o', s=100, facecolors='none', edgecolors=col)
                ax.annotate(f'{i}', xy=ri, color=col, xytext=(5,5), textcoords="offset points")
                ax.annotate(f'{j}', xy=ri+dist, color=col, xytext=(5,-10), textcoords="offset points")
                Dvec = np.cross(D, dist)
                Dvec = Dvec/np.linalg.norm(Dvec)
                ax.quiver(*(ri+dist/2.0), Dvec[0], Dvec[1], units='xy', scale=10, zorder=-1,color='gray')
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            if abs(Dvec[k]*self.epsilon(k,l,m)) > 0 and np.linalg.norm(dist) >= 0.9:
                                if pt and model_params['bc_classical']:
                                    if m==2: self.add_coupling_term(Dvec[k]*self.epsilon(k,l,m)/2.0, i, j, Svec[l], "Id")
                                    if l==2: self.add_coupling_term(Dvec[k]*self.epsilon(k,l,m)/2.0, i, j, "Id", Svec[m])
                                else:
                                    self.add_coupling_term(Dvec[k]*self.epsilon(k,l,m), i, j, Svec[l], Svec[m])
                if pt and model_params['bc_classical']:
                    self.add_coupling_term(J[2]/2, i, j, "Sz", "Id")
                    self.add_coupling_term(J[2]/2, i, j, "Id", "Sz")
                else:
                    for (Ji, Si) in zip(J, Svec):
                        self.add_coupling_term(Ji, i, j, Si, Si)
                ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('latt.png', dpi=600)


class MySpinChain(chiral_magnet, NearestNeighborModel):
    """The :class:`chiral_magnet` on a Chain, suitable for TEBD.

    See the :class:`chiral_magnet` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

def measurements(results, psi, model, simulation, tol=0.01):
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

class my_triangular(SimpleLattice):
    """A triangular lattice.

    .. plot ::


        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(6, 8))
        ax = plt.gca()
        lat = lattice.Triangular(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linestyle='-', linewidth=3, label='nearest_neighbors')
        for key, lw in zip(['next_nearest_neighbors',
                            'next_next_nearest_neighbors'],
                        [1.5, 1.]):
            pairs = lat.pairs[key]
            lat.plot_coupling(ax, pairs, linestyle='--', linewidth=lw, color='gray', label=key)
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        ax.legend(loc='upper left')
        plt.show()

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
        lat = lattice.Triangular(4, 4, None, bc='periodic')
        order_names=['default', 'snake']
        for order_name, ax in zip(order_names, axes.flatten()):
            lat.plot_coupling(ax, linestyle='-', linewidth=1)
            lat.order = lat.ordering(order_name)
            lat.plot_order(ax, linestyle=':', linewidth=2)
            lat.plot_sites(ax)
            lat.plot_basis(ax, origin=(-0.5, -0.5))
            ax.set_title(f"order={order_name!r}")
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 6)
            ax.set_aspect('equal')
        plt.show()


    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `pairs` are set accordingly.
        If `order` is specified in the form ``('standard', snake_winding, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, site, **kwargs):
        sqrt3_half = 0.5 * np.sqrt(3)  # = cos(pi/6)
        basis = np.array([[1., 0.], [0.5, sqrt3_half]])
        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([-1, 1])), (0, 0, np.array([0, -1]))]
        nNN = [(0, 0, np.array([2, -1])), (0, 0, np.array([1, 1])), (0, 0, np.array([-1, 2]))]
        nnNN = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2])), (0, 0, np.array([-2, 2]))]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        SimpleLattice.__init__(self, [Lx, Ly], site, **kwargs)

class my_square(SimpleLattice):
    """Simple square lattice.

    .. plot ::


        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(6, 8))
        ax = plt.gca()
        lat = lattice.Triangular(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linestyle='-', linewidth=3, label='nearest_neighbors')
        for key, lw in zip(['next_nearest_neighbors',
                            'next_next_nearest_neighbors'],
                        [1.5, 1.]):
            pairs = lat.pairs[key]
            lat.plot_coupling(ax, pairs, linestyle='--', linewidth=lw, color='gray', label=key)
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        ax.legend(loc='upper left')
        plt.show()

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
        lat = lattice.Triangular(4, 4, None, bc='periodic')
        order_names=['default', 'snake']
        for order_name, ax in zip(order_names, axes.flatten()):
            lat.plot_coupling(ax, linestyle='-', linewidth=1)
            lat.order = lat.ordering(order_name)
            lat.plot_order(ax, linestyle=':', linewidth=2)
            lat.plot_sites(ax)
            lat.plot_basis(ax, origin=(-0.5, -0.5))
            ax.set_title(f"order={order_name!r}")
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 6)
            ax.set_aspect('equal')
        plt.show()


    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `pairs` are set accordingly.
        If `order` is specified in the form ``('standard', snake_winding, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, site, **kwargs):
        basis = np.array([[1., 0.], [0., 1.]])
        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        SimpleLattice.__init__(self, [Lx, Ly], site, **kwargs)