from tenpy.models.lattice import SimpleLattice
import numpy as np

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