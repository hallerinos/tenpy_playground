import warnings
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPSEnvironment

def concurrence(mps, ops1, ops2, sites1=None, sites2=None, opstr=None, str_on_first=True, hermitian=False, autoJW=True):
        r"""Correlation function  ``<bra|op1_i op2_j|ket>`` of single site operators,
        sandwiched between bra and ket.
        For examples the contraction for a two-site operator on site `i` would look like::

            |          .--S--B[i]--B[i+1]--...--B[j]---.
            |          |     |     |            |      |
            |          |     |     |            op2    |
            |          LP[i] |     |            |      RP[j]
            |          |     op1   |            |      |
            |          |     |     |            |      |
            |          .--S--B*[i]-B*[i+1]-...--B*[j]--.


        Onsite terms are taken in the order ``<psi | op1 op2 | psi>``.
        If `opstr` is given and ``str_on_first=True``, it calculates::

            |           for i < j                               for i > j
            |
            |          .--S--B[i]---B[i+1]--...- B[j]---.     .--S--B[j]---B[j+1]--...- B[i]---.
            |          |     |      |            |      |     |     |      |            |      |
            |          |     opstr  opstr        op2    |     |     op2    |            |      |
            |          LP[i] |      |            |      RP[j] LP[j] |      |            |      RP[i]
            |          |     op1    |            |      |     |     opstr  opstr        op1    |
            |          |     |      |            |      |     |     |      |            |      |
            |          .--S--B*[i]--B*[i+1]-...- B*[j]--.     .--S--B*[j]--B*[j+1]-...- B*[i]--.


        For ``i==j``, no `opstr` is included.
        For ``str_on_first=False``, the `opstr` on site ``min(i, j)`` is always left out.
        Strings (like ``'Id', 'Sz'``) in the arguments are translated into single-site
        operators defined by the :class:`~tenpy.networks.site.Site` on which they act.
        Each operator should have the two legs ``'p', 'p*'``.

        .. warning ::
            This function is only evaluating correlation functions by moving right, and hence
            can be inefficient if you try to vary the left end while fixing the right end.
            In that case, you might be better off (=faster evaluation) by using
            :meth:`term_correlation_function_left` with a small for loop over the right indices.

        Parameters
        ----------
        ops1 : (list of) { :class:`~tenpy.linalg.np_conserved.Array` | str }
            First operator of the correlation function (acting after ops2).
            If a list is given, ``ops1[i]`` acts on site `i` of the MPS.
            Note that even if a list is given, we still just evaluate two-site correlations!
            ``psi.correlation_function(['A','B'], ['C', 'D'])`` evaluates
            ``<A_i C_j>`` for even i and even j, ``<B_i C_j>`` for even i and odd j,
            ``<B_i C_j>`` for odd i and even j, and ``<B_i D_j>`` for odd i and odd j.
        ops2 : (list of) { :class:`~tenpy.linalg.np_conserved.Array` | str }
            Second operator of the correlation function (acting before ops1).
            If a list is given, ``ops2[j]`` acts on site `j` of the MPS.
        sites1 : None | int | list of int
            List of site indices `i`; a single `int` is translated to ``range(0, sites1)``.
            ``None`` defaults to all sites ``range(0, L)``.
            Is sorted before use, i.e. the order is ignored.
        sites2 : None | int | list of int
            List of site indices; a single `int` is translated to ``range(0, sites2)``.
            ``None`` defaults to all sites ``range(0, L)``.
            Is sorted before use, i.e. the order is ignored.
        opstr : None | (list of) { :class:`~tenpy.linalg.np_conserved.Array` | str }
            Ignored by default (``None``).
            Operator(s) to be inserted between ``ops1`` and ``ops2``.
            If less than :attr:`L` operators are given, we repeat them periodically.
            If given as a list, ``opstr[r]`` is inserted at site `r` (independent of `sites1` and
            `sites2`).
        str_on_first : bool
            Whether the `opstr` is included on the site ``min(i, j)``.
            Note the order, which is chosen that way to handle fermionic Jordan-Wigner strings
            correctly. (In other words: choose ``str_on_first=True`` for fermions!)
        hermitian : bool
            Optimization flag: if ``sites1 == sites2`` and ``Ops1[i]^\dagger == Ops2[i]``
            (which is not checked explicitly!), the resulting ``C[x, y]`` will be hermitian.
            We can use that to avoid calculations, so ``hermitian=True`` will run faster.
        autoJW : bool
            *Ignored* if `opstr` is given.
            If `True`, auto-determine if a Jordan-Wigner string is needed.
            Works only if exclusively strings were used for `op1` and `op2`.

        Returns
        -------
        C : 2D ndarray
            The correlation function ``C[x, y] = <bra|ops1[i] ops2[j]|ket>``,
            where ``ops1[i]`` acts on site ``i=sites1[x]`` and ``ops2[j]`` on site ``j=sites2[y]``.
            If `opstr` is given, it gives (for ``str_on_first=True``):
            - For ``i < j``: ``C[x, y] = <bra|ops1[i] prod_{i <= r < j} opstr[r] ops2[j]|ket>``.
            - For ``i > j``: ``C[x, y] = <bra|prod_{j <= r < i} opstr[r] ops1[i] ops2[j]|ket>``.
            - For ``i = j``: ``C[x, y] = <bra|ops1[i] ops2[j]|ket>``.
            The condition ``<= r`` is replaced by a strict ``< r``, if ``str_on_first=False``.

            .. warning ::

                The :class:`MPSEnvironment` variant of this method takes the accumulated MPS
                :attr:`~tenpy.networks.mps.MPS.norm` into account, which is non-trivial e.g. when you
                used `apply_local_op` with non-unitary operators.

                In contrast, the :class:`MPS` variant of this method *ignores* the `norm`,
                i.e. returns the expectation value for the normalized state.

        Examples
        --------
        Let's prepare a state in alternating ``|+z>, |+x>`` states:

        .. doctest :: MPS.correlation_function

            >>> spin_half = tenpy.networks.site.SpinHalfSite(conserve=None)
            >>> p_state = ['up', [np.sqrt(0.5), -np.sqrt(0.5)]]*3
            >>> psi = tenpy.networks.mps.MPS.from_product_state([spin_half]*6, p_state, "infinite")

        Default arguments calculate correlations for all `i` and `j` within the MPS unit cell.
        To evaluate the correlation function for a single `i`, you can use ``sites1=[i]``.
        Alternatively, you can use :meth:`term_correlation_function_right`
        (or :meth:`term_correlation_function_left`):

        .. doctest :: MPS.correlation_function

            >>> psi.correlation_function("Sz", "Sx")  # doctest: +SKIP
            array([[ 0.  , -0.25,  0.  , -0.25,  0.  , -0.25],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                   [ 0.  , -0.25,  0.  , -0.25,  0.  , -0.25],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                   [ 0.  , -0.25,  0.  , -0.25,  0.  , -0.25],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
            >>> psi.correlation_function("Sz", "Sx", [0])
            array([[ 0.  , -0.25,  0.  , -0.25,  0.  , -0.25]])
            >>> corr1 = psi.correlation_function("Sz", "Sx", [0], range(1, 10))
            >>> corr2 = psi.term_correlation_function_right([("Sz", 0)], [("Sx", 0)], 0, range(1, 10))
            >>> assert np.all(np.abs(corr2 - corr1) < 1.e-12)

        For fermions, it auto-determines that/whether a Jordan Wigner string is needed:

        .. doctest :: MPS.correlation_function

            >>> fermion = tenpy.networks.site.FermionSite(conserve='N')
            >>> p_state = ['empty', 'full'] * 3
            >>> psi = tenpy.networks.mps.MPS.from_product_state([fermion]*6, p_state, "finite")
            >>> CdC = psi.correlation_function("Cd", "C")  # optionally: use `hermitian=True`
            >>> psi.correlation_function("C", "Cd")[1, 2] == -CdC[2, 1]
            True
            >>> np.all(np.diag(CdC) == psi.expectation_value("Cd C"))  # "Cd C" is equivalent to "N"
            True

        See also
        --------
        expectation_value_term : for a single combination of `i` and `j` of ``A_i B_j```.
        term_correlation_function_right : for correlations between multi-site terms, fix left term.
        term_correlation_function_left : for correlations between multi-site terms, fix right term.
        """
        if opstr is not None:
            autoJW = False
        ops1, ops2, sites1, sites2, opstr = mps._correlation_function_args(
            ops1, ops2, sites1, sites2, opstr)
        if ((len(sites1) > 2 * len(sites2) and min(sites2) > max(sites1) - len(sites2))
                or (len(sites2) > 2 * len(sites1) and min(sites1) > max(sites2) - len(sites1))):
            warnings.warn(
                "Inefficient evaluation of MPS.correlation_function(), "
                "it's probably faster to use MPS.term_correlation_function_left()",
                stacklevel=2)
        if autoJW and not all([isinstance(op1, str) for op1 in ops1]):
            warnings.warn("Non-string operator: can't auto-determine Jordan-Wigner!", stacklevel=2)
            autoJW = False
        if autoJW:
            need_JW = []
            for i in sites1:
                need_JW.append(mps.sites[i % mps.L].op_needs_JW(ops1[i % len(ops1)]))
            for j in sites2:
                need_JW.append(mps.sites[j % mps.L].op_needs_JW(ops1[j % len(ops1)]))
            if any(need_JW):
                if not all(need_JW):
                    raise ValueError("Some, but not any operators need 'JW' string!")
                if not str_on_first:
                    raise ValueError("Need Jordan Wigner string, but `str_on_first`=False`")
                opstr = ['JW']
        if hermitian and np.any(sites1 != sites2):
            warnings.warn("MPS correlation function can't use the hermitian flag", stacklevel=2)
            hermitian = False
        C = np.empty((len(sites1), len(sites2)), dtype=complex)
        for x, i in enumerate(sites1):
            # j > i
            j_gtr = sites2[sites2 > i]
            if len(j_gtr) > 0:
                C_gtr = _my_corr_up_diag(mps, ops1, ops2, i, j_gtr, opstr, str_on_first, True)
                C[x, (sites2 > i)] = C_gtr
                if hermitian:
                    C[x + 1:, x] = np.conj(C_gtr)
            # j == i
            j_eq = sites2[sites2 == i]
            if len(j_eq) > 0:
                # on-site correlation function
                op1, _ = mps.get_op(ops1, i)
                op2, _ = mps.get_op(ops2, i)
                op12 = npc.tensordot(op1, op2, axes=['p*', 'p'])
                C[x, (sites2 == i)] = my_expectation_value(mps,op12, i, [['p'], ['p*']])
        if not hermitian:
            #  j < i
            for y, j in enumerate(sites2):
                i_gtr = sites1[sites1 > j]
                if len(i_gtr) > 0:
                    C[(sites1 > j), y] = _my_corr_up_diag(mps, ops2, ops1, j, i_gtr, opstr,
                                                            str_on_first, False)
                    # exchange ops1 and ops2 : they commute on different sites,
                    # but we apply opstr after op1 (using the last argument = False)
        return mps._normalize_exp_val(C)

def _my_corr_up_diag(mps, ops1, ops2, i, j_gtr, opstr, str_on_first, apply_opstr_first):
        """correlation function above the diagonal: for fixed i and all j in j_gtr, j > i."""
        op1, _ = mps.get_op(ops1, i)
        opstr1, _ = mps.get_op(opstr, i)
        if opstr1 is not None and str_on_first:
            axes = ['p*', 'p'] if apply_opstr_first else ['p', 'p*']
            op1 = npc.tensordot(op1, opstr1, axes=axes)
        bra, ket = mps._get_bra_ket()
        theta_ket = ket.get_B(i, form='Th')
        theta_bra = bra.get_B(i, form='Th')
        C = npc.tensordot(op1, theta_ket, axes=['p*', 'p'])
        C = mps._contract_with_LP(C, i)
        axes_contr = [['vL*'] + ket._get_p_label('*'), ['vR*'] + ket._p_label]
        C = npc.tensordot(theta_bra.conj(), C, axes=axes_contr)
        # C has legs 'vR*', 'vR'
        js = list(j_gtr[::-1])  # stack of j, sorted *descending*
        res = []
        for r in range(i + 1, js[0] + 1):  # js[0] is the maximum
            B_ket = ket.get_B(r, form='B')
            B_bra = bra.get_B(r, form='B')
            C = npc.tensordot(C, B_ket.complex_conj(), axes=['vR', 'vL'])  # CHANGED ONLY THIS LINE
            if r == js[-1]:
                op2, _ = mps.get_op(ops2, r)
                Cij = npc.tensordot(op2, C, axes=['p*', 'p'])
                Cij = mps._contract_with_RP(Cij, r)
                Cij.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                Cij = npc.inner(B_bra.conj(), Cij, axes='labels')
                res.append(Cij)
                js.pop()
            if len(js) > 0:
                op, _ = mps.get_op(opstr, r)
                if op is not None:
                    C = npc.tensordot(op, C, axes=['p*', 'p'])
                C = npc.tensordot(B_bra.conj(), C, axes=axes_contr)
        return res

def my_expectation_value(mps, ops, sites=None, axes=None):
        """Expectation value ``<bra|ops|ket>`` of (n-site) operator(s).

        Calculates n-site expectation values of operators sandwiched between bra and ket.
        For examples the contraction for a two-site operator on site `i` would look like::

            |          .--S--B[i]--B[i+1]--.
            |          |     |     |       |
            |          |     |-----|       |
            |          LP[i] | op  |       RP[i+1]
            |          |     |-----|       |
            |          |     |     |       |
            |          .--S--B*[i]-B*[i+1]-.

        Here, the `B` are taken from `ket`, the `B*` from `bra`.
        For MPS expectation values these are the same and LP/ RP are trivial.


        Parameters
        ----------
        ops : (list of) { :class:`~tenpy.linalg.np_conserved.Array` | str }
            The operators, for which the expectation value should be taken,
            All operators should all have the same number of legs (namely `2 n`).
            If less than ``len(sites)`` operators are given, we repeat them periodically.
            Strings (like ``'Id', 'Sz'``) are translated into single-site operators defined by
            :attr:`sites`.
        sites : list
            List of site indices. Expectation values are evaluated there.
            If ``None`` (default), the entire chain is taken (clipping for finite b.c.)
        axes : None | (list of str, list of str)
            Two lists of each `n` leg labels giving the physical legs of the operator used for
            contraction. The first `n` legs are contracted with conjugated `B`,
            the second `n` legs with the non-conjugated `B`.
            ``None`` defaults to ``(['p'], ['p*'])`` for single site (n=1), or
            ``(['p0', 'p1', ... 'p{n-1}'], ['p0*', 'p1*', .... 'p{n-1}*'])`` for `n` > 1.

        Returns
        -------
        exp_vals : 1D ndarray
            Expectation values, ``exp_vals[i] = <bra|ops[i]|ket>``, where ``ops[i]`` acts on
            site(s) ``j, j+1, ..., j+{n-1}`` with ``j=sites[i]``.

            .. warning ::

                The :class:`MPSEnvironment` variant of this method takes the accumulated MPS
                :attr:`~tenpy.networks.mps.MPS.norm` into account, which is non-trivial e.g. when you
                used `apply_local_op` with non-unitary operators.

                In contrast, the :class:`MPS` variant of this method *ignores* the `norm`,
                i.e. returns the expectation value for the normalized state.

        Examples
        --------
        Let's prepare a state in alternating ``|+z>, |+x>`` states:

        .. doctest :: MPS.expectation_value

            >>> spin_half = tenpy.networks.site.SpinHalfSite(conserve=None)
            >>> p_state = ['up', [np.sqrt(0.5), -np.sqrt(0.5)]]*3
            >>> psi = tenpy.networks.mps.MPS.from_product_state([spin_half]*6, p_state)

        One site examples (n=1):

        .. doctest :: MPS.expectation_value

            >>> Sz = psi.expectation_value('Sz')
            >>> print(Sz)
            [0.5 0.  0.5 0.  0.5 0. ]
            >>> Sx = psi.expectation_value('Sx')
            >>> print(Sx)
            [ 0.  -0.5  0.  -0.5  0.  -0.5]
            >>> print(psi.expectation_value(['Sz', 'Sx']))
            [ 0.5 -0.5  0.5 -0.5  0.5 -0.5]
            >>> print(psi.expectation_value('Sz', sites=[0, 3, 4]))
            [0.5 0.  0.5]

        Two site example (n=2), assuming homogeneous sites:

        .. doctest :: MPS.expectation_value

            >>> SzSx = npc.outer(psi.sites[0].Sz.replace_labels(['p', 'p*'], ['p0', 'p0*']),
            ...                  psi.sites[1].Sx.replace_labels(['p', 'p*'], ['p1', 'p1*']))
            >>> print(psi.expectation_value(SzSx))  # note: len L-1 for finite bc, or L for infinite
            [-0.25  0.   -0.25  0.   -0.25]

        Example measuring <psi|SzSx|psi> on each second site, for inhomogeneous sites:

        .. doctest :: MPS.expectation_value

            >>> SzSx_list = [npc.outer(psi.sites[i].Sz.replace_labels(['p', 'p*'], ['p0', 'p0*']),
            ...                        psi.sites[i+1].Sx.replace_labels(['p', 'p*'], ['p1', 'p1*']))
            ...              for i in range(0, psi.L-1, 2)]
            >>> print(psi.expectation_value(SzSx_list, range(0, psi.L-1, 2)))
            [-0.25 -0.25 -0.25]

        Expectation value with different bra and ket in an MPSEnvironment:

        .. doctest :: MPS.expectation_value

            >>> spin_half = tenpy.networks.site.SpinHalfSite(conserve=None)
            >>> p2_state = [[np.sqrt(0.5), -np.sqrt(0.5)], 'up']*3
            >>> phi = tenpy.networks.mps.MPS.from_product_state([spin_half]*6, p2_state)
            >>> env = tenpy.networks.mps.MPSEnvironment(phi, psi)
            >>> Sz = env.expectation_value('Sz')
            >>> print(Sz)
            [0.0625 0.0625 0.0625 0.0625 0.0625 0.0625]

        """
        ops, sites, n, (op_ax_p, op_ax_pstar) = mps._expectation_value_args(ops, sites, axes)
        ax_p = ['p' + str(k) for k in range(n)]
        ax_pstar = ['p' + str(k) + '*' for k in range(n)]
        bra, ket = mps._get_bra_ket()
        E = []
        for i in sites:
            op, needs_JW = mps.get_op(ops, i)
            op = op.replace_labels(op_ax_p + op_ax_pstar, ax_p + ax_pstar)
            theta_ket = ket.get_theta(i, n).complex_conj()  # changed this line!!!
            if needs_JW:
                if isinstance(mps, MPSEnvironment):
                    mps.apply_JW_string_left_of_virt_leg(theta_ket, 'vL', i)
                else:
                    msg = "Expectation value of operator that needs JW string can't work"
                    raise ValueError(msg)
            C = npc.tensordot(op, theta_ket, axes=[ax_pstar, ax_p])  # C has same labels as theta
            C = mps._contract_with_LP(C, i)  # axes_p + (vR*, vR)
            C = mps._contract_with_RP(C, i + n - 1)  # axes_p + (vR*, vL*)
            C.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])  # back to original theta labels
            theta_bra = bra.get_theta(i, n)
            E.append(npc.inner(theta_bra, C, axes='labels', do_conj=True))
        return mps._normalize_exp_val(E)