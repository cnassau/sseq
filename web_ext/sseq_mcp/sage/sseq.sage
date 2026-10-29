# sseq: minimal resolutions over the Steenrod algebra, computed by the sseq MCP server.
#
# This file is sent to a mcp-sage workspace and run there, so it must pass mcp-sage's sandbox: no
# imports, no dunder names anywhere (hence `setup` in place of a constructor), and none of the
# persistence calls (hence `add_data` rather than `load`). Sage code cannot reach the sseq server;
# data arrives as Sage code returned by the `sseq_resolution_data` tool.


class SseqMissingData(ValueError):
    """
    Raised when the resolution is asked about a bidegree it holds no data for.

    The message names the `sseq_resolution_data` call that supplies the data.
    """
    pass


# Basis keys `(t, i)` of every `C_s`: the `i`-th minimal generator in internal degree `t`.
SSEQ_KEYS = cartesian_product([NonNegativeIntegers(), NonNegativeIntegers()])


class SseqFreeModule(CombinatorialFreeModule):
    """
    A free graded module over a Steenrod algebra on generators `x[(t, i)]` of degree `t`.

    The degree of `a * x[(t, i)]` is `deg(a) + t`. Sage's graded modules with basis take the
    degree of an element from its basis keys alone, as if the coefficients had degree zero, so the
    element methods that depend on degrees are redefined here to count the coefficients' degrees.
    """

    def degree_on_basis(self, key):
        """The degree `t` of the generator `x[(t, i)]`."""
        return key[0]

    def basis(self, d=None):
        """
        The basis of generators; `basis(d)` is not available.

        The keys run over all pairs `(t, i)`, so the generators of one degree cannot be found by
        searching them; `SseqResolution.gens(s, t)` lists them from the resolution's data.
        """
        if d is not None:
            raise NotImplementedError(
                "use R.gens(s, t) for the generators of C_s in degree t, or R.degree_basis(s, t) "
                "for a basis of the degree t part over the ground field"
            )
        return CombinatorialFreeModule.basis(self)

    class Element(CombinatorialFreeModule.Element):
        """An element of a `SseqFreeModule`, graded by `deg(a) + t` on `a * x[(t, i)]`."""

        def homogeneous_components(self):
            """A dict from degrees to the nonzero homogeneous components of this element."""
            P = self.parent()
            A = P.base_ring()
            terms = {}
            for key, c in self:
                for m, k in c.monomial_coefficients().items():
                    a = k * A.monomial(m)
                    terms.setdefault(a.degree() + key[0], []).append((key, a))
            return {d: P.sum_of_terms(ts) for d, ts in terms.items()}

        def is_homogeneous(self):
            """Whether this element lies in a single degree; zero does."""
            return len(self.homogeneous_components()) <= 1

        def degree(self):
            """The degree of this element, which must be homogeneous and nonzero."""
            components = self.homogeneous_components()
            if not components:
                raise ValueError("the zero element does not have a well-defined degree")
            if len(components) > 1:
                raise ValueError("element is not homogeneous")
            return next(iter(components))

        homogeneous_degree = degree

        def maximal_degree(self):
            """The largest degree of a homogeneous component of this nonzero element."""
            components = self.homogeneous_components()
            if not components:
                raise ValueError("the zero element does not have a well-defined degree")
            return max(components)

        def homogeneous_component(self, n):
            """The degree `n` part of this element."""
            return self.homogeneous_components().get(n, self.parent().zero())

        def truncate(self, n):
            """The sum of the homogeneous components of degree less than `n`."""
            components = self.homogeneous_components()
            return self.parent().sum(c for d, c in components.items() if d < n)


class SseqResolution(SageObject):
    """
    A minimal free resolution `... -> C_2 -> C_1 -> C_0` of a module over a Steenrod algebra.

    Each `C_s` is a graded free module over `self.algebra()` (a `SseqFreeModule`) whose basis is the
    set of minimal generators, and the differentials preserve degrees. Generators and differentials
    become known as data is added with `add_data`; asking about anything else raises
    `SseqMissingData`.
    """

    def setup(self, rid, p, algebra):
        """Initialise an empty resolution; use `sseq_resolution` instead of calling this."""
        self._rid = rid
        self._p = p
        self._A = algebra
        # s -> largest t for which the generators and differentials of `C_s` are complete.
        self._t_max = {}
        # (s, t) -> number of generators; absent means zero.
        self._ngens = {}
        # (s, t, i) -> terms of the differential of that generator.
        self._diff = {}
        self._C = {}
        self._d = {}

    def _repr_(self):
        return "Minimal resolution %s over %s" % (self._rid, self._A)

    def id(self):
        """The resolution's id at the sseq server."""
        return self._rid

    def prime(self):
        """The prime `p`."""
        return self._p

    def algebra(self):
        """The Steenrod algebra (or sub-Hopf-algebra) the resolution is over."""
        return self._A

    def add_data(self, s_range, t_max, ngens, diffs):
        """
        Record data sent by `sseq_resolution_data`; not meant to be called by hand.

        The data is complete for `s_range[0] <= s <= s_range[1]` and `t <= t_max`: `ngens` maps
        `(s, t)` to a nonzero generator count and `diffs` maps `(s, t, i)` to the differential of
        that generator, a list of terms `(milnor_key, (t', j))` or, at odd primes,
        `(coefficient, milnor_key, (t', j))`.
        """
        self._ngens.update(ngens)
        self._diff.update(diffs)
        for s in range(s_range[0], s_range[1] + 1):
            self._t_max[s] = max(self._t_max.get(s, -1), t_max)

    def computed_range(self):
        """A dict `s -> t_max` saying which bidegrees hold data."""
        return dict(self._t_max)

    def has_data(self, s, t):
        """Whether data for bidegree `(s, t)` has been added."""
        return t <= self._t_max.get(s, -1)

    def _need(self, s, t):
        if not self.has_data(s, t):
            raise SseqMissingData(
                "%s has no data for (s, t) = (%s, %s). Call the sseq MCP tool "
                "sseq_resolution_data with id=%r, s_min=%s, s_max=%s, t_max=%s and run the code "
                "it returns in this workspace." % (self._rid, s, t, self._rid, s, s, t)
            )

    def ngens(self, s, t):
        """The number of minimal generators of `C_s` in internal degree `t`."""
        if s < 0:
            return 0
        self._need(s, t)
        return self._ngens.get((s, t), 0)

    def C(self, s):
        """The graded free module `C_s`, with basis keys `(t, i)` of degree `t`."""
        if s not in self._C:
            self._C[s] = SseqFreeModule(
                self._A,
                SSEQ_KEYS,
                prefix="x%s" % s,
                category=ModulesWithBasis(self._A).Graded(),
            )
        return self._C[s]

    def gen(self, s, t, i):
        """The `i`-th generator of `C_s` in internal degree `t`."""
        if not 0 <= i < self.ngens(s, t):
            raise IndexError("C_%s has %s generators in degree %s" % (s, self.ngens(s, t), t))
        return self.C(s).monomial(SSEQ_KEYS((t, i)))

    def gens(self, s, t):
        """The generators of `C_s` in internal degree `t`."""
        return [self.gen(s, t, i) for i in range(self.ngens(s, t))]

    def d(self, s):
        """The differential `C_s -> C_{s-1}`, an algebra-linear map preserving degrees."""
        if s not in self._d:
            A = self._A
            target = self.C(s - 1)
            diff = self._diff

            def on_basis(key):
                t, i = key
                if not 0 <= i < self.ngens(s, t):
                    raise IndexError("C_%s has no generator %s" % (s, key))
                terms = []
                for term in diff.get((s, t, i), []):
                    if len(term) == 2:
                        c, m, g = 1, term[0], term[1]
                    else:
                        c, m, g = term
                    terms.append((SSEQ_KEYS(g), c * A.monomial(m)))
                return target.sum_of_terms(terms)

            self._d[s] = self.C(s).module_morphism(on_basis=on_basis, codomain=target)
        return self._d[s]

    def degree_basis(self, s, t):
        """A basis of the `F_p`-vector space `(C_s)_t`: operations times generators."""
        return [
            a * self.C(s).monomial(SSEQ_KEYS((u, i)))
            for u in range(t + 1)
            for i in range(self.ngens(s, u))
            for a in self._A.basis(t - u)
        ]

    def ext_dims(self, s_max, t_max):
        """
        A dict `(n, s) -> dim Ext^{s, n+s}` for the nonzero groups with `s <= s_max`, `t <= t_max`.

        The resolution is minimal, so these are the generator counts.
        """
        return {
            (t - s, s): self.ngens(s, t)
            for s in range(s_max + 1)
            for t in range(t_max + 1)
            if self.ngens(s, t)
        }

    def check_d_squared(self, s_max, t_max):
        """The generators `x` with `s <= s_max`, `t <= t_max` and `d(d(x)) != 0`; empty if none."""
        return [
            (s, t, i)
            for s in range(2, s_max + 1)
            for t in range(t_max + 1)
            for i in range(self.ngens(s, t))
            if self.d(s - 1)(self.d(s)(self.gen(s, t, i))) != 0
        ]


def sseq_resolution(rid, p, algebra):
    """A resolution with no data yet; the `sseq_resolve` tool returns the call to make."""
    R = SseqResolution()
    R.setup(rid, p, algebra)
    return R
