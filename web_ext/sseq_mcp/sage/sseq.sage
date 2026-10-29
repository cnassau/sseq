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


class SseqModule(CombinatorialFreeModule):
    """
    A module over a Steenrod algebra, as a graded `F_p`-vector space with the algebra acting.

    Subclasses say how a Milnor basis element acts on a basis element (`_act_on_basis`) and how
    basis elements are named. A module may be truncated: it then holds degrees up to `t_max` only,
    and acting beyond raises `SseqMissingData`; a complete module is zero above `t_max`.

    `a * x` works for `a` in `self.algebra()`; `self.act(a, x)` also accepts any element that
    converts into it.
    """

    def _repr_(self):
        truncated = "" if self._complete else ", truncated above degree %s" % self._t_max
        return "Steenrod module %s over %s%s" % (self._module_name, self._A, truncated)

    def algebra(self):
        """The Steenrod algebra (or sub-Hopf-algebra) acting on the module."""
        return self._A

    def module_name(self):
        """The module's name."""
        return self._module_name

    def t_max(self):
        """The largest degree the module holds."""
        return self._t_max

    def is_complete(self):
        """Whether the module is all there, rather than truncated at `t_max`."""
        return self._complete

    def bottom(self):
        """The lowest degree of a basis element."""
        return min(self.degree_on_basis(key) for key in self.basis().keys())

    def basis_in_degree(self, d):
        """The basis elements of degree `d`."""
        return [self.monomial(k) for k in self.basis().keys() if self.degree_on_basis(k) == d]

    def act(self, a, x):
        """The action `a * x`, for `a` anything that converts into `self.algebra()`."""
        A = self._A
        try:
            a = A(a)
        except (TypeError, ValueError):
            raise ValueError("%s is not in %s" % (a, A))
        terms = []
        for key, c in x:
            t = self.degree_on_basis(key)
            for m, k in a.monomial_coefficients().items():
                d = A.monomial(m).degree()
                if d == 0:
                    terms.append((key, c * k))
                    continue
                if t + d > self._t_max:
                    if self._complete:
                        continue
                    raise SseqMissingData(
                        "%s is truncated above degree %s; %s"
                        % (self._module_name, self._t_max, self._truncation_hint)
                    )
                for target, coeff in self._act_on_basis(m, key):
                    terms.append((target, c * k * coeff))
        return self.sum_of_terms(terms)

    def submodule_basis(self, gens):
        """
        A dict from degrees to bases of the submodule generated by the homogeneous elements `gens`.

        For a truncated module this covers degrees up to `t_max`.
        """
        A = self._A
        spans = {}
        for g in gens:
            if g == 0:
                continue
            e = g.degree()
            for d in range(e, self._t_max + 1):
                spans.setdefault(d, []).extend(self.act(b, g) for b in A.basis(d - e))
        result = {}
        for d, elements in sorted(spans.items()):
            keys = [k for k in self.basis().keys() if self.degree_on_basis(k) == d]
            rows = [vector(self.base_ring(), [v[k] for k in keys]) for v in elements]
            echelon = matrix(self.base_ring(), len(rows), len(keys), rows).echelon_form()
            basis = [
                self.sum_of_terms((k, c) for k, c in zip(keys, row) if c)
                for row in echelon.rows()
                if row
            ]
            if basis:
                result[d] = basis
        return result

    def submodule(self, gens, name=None):
        """
        The submodule generated by the homogeneous elements `gens`, as a module in its own right.

        Its basis is the echelon basis of `submodule_basis(gens)`, named `e<d>` or `e<d>_<i>` by
        degree. It is truncated where this module is.
        """
        A = self._A
        F = self.base_ring()
        basis = self.submodule_basis(gens)
        keys = {
            d: [k for k in self.basis().keys() if self.degree_on_basis(k) == d] for d in basis
        }

        def coordinates(v, d):
            return vector(F, [v[k] for k in keys[d]])

        rows = {d: matrix(F, [coordinates(v, d) for v in vs]) for d, vs in basis.items()}
        names = {}
        actions = {}
        for d, vs in basis.items():
            for i, v in enumerate(vs):
                names[(d, i)] = "e%s_%s" % (d, i) if len(vs) > 1 else "e%s" % d
                for e in range(1, self._t_max - d + 1):
                    for b in A.basis(e):
                        w = self.act(b, v)
                        if w == 0:
                            continue
                        solution = rows[d + e].solve_left(coordinates(w, d + e))
                        m = list(b.monomial_coefficients())[0]
                        actions[(m, (d, i))] = [
                            (c, (d + e, j)) for j, c in enumerate(solution) if c
                        ]
        if name is None:
            name = "submodule of %s" % self._module_name
        return sseq_steenrod_module(
            name, A.prime(), A, names, self._t_max, self._complete, actions
        )

    def spec(self):
        """
        This module as a module spec (the JSON format of ext/MODULE-SPEC.md), as a string.

        `sseq_resolve` resolves it when given the spec and a name for it. Only complete modules
        have one. The basis is renamed `x<t>_<i>`, since the spec format reads names up to spaces.
        """
        if not self._complete:
            raise ValueError("%s is truncated, so it has no spec" % self._module_name)
        A = self._A
        p = A.prime()
        keys = sorted(self.basis().keys(), key=lambda k: (self.degree_on_basis(k), k))
        names = {}
        count = {}
        for k in keys:
            t = self.degree_on_basis(k)
            names[k] = "x%s_%s" % (t, count.get(t, 0))
            count[t] = count.get(t, 0) + 1
        span = self._t_max - self.bottom()
        # The generators of the algebra, by the names the spec format gives them.
        operations = []
        if p == 2:
            n = 1
            while n <= span:
                operations.append(("Sq%s" % n, (n,)))
                n *= 2
        else:
            operations.append(("b", ((0,), ())))
            n = 1
            while 2 * (p - 1) * n <= span:
                operations.append(("P%s" % n, ((), (n,))))
                n *= p
        actions = []
        full = SteenrodAlgebra(p, basis="milnor")
        for label, m in operations:
            # Converting from the whole algebra checks that the generator is in this one.
            try:
                a = A(full.monomial(m))
            except (TypeError, ValueError):
                continue
            for k in keys:
                w = self.act(a, self.monomial(k))
                if w == 0:
                    continue
                terms = " + ".join(
                    ("" if c == 1 else "%s " % c) + names[j] for j, c in w
                )
                actions.append('"%s %s = %s"' % (label, names[k], terms))
        gens = ", ".join('"%s": %s' % (names[k], self.degree_on_basis(k)) for k in keys)
        algebra = ""
        if A._has_nontrivial_profile():
            truncated = A._truncation_type == 0
            if p == 2:
                p_part, q_part = list(A._profile), None
            else:
                p_part = list(A._profile[0])
                k = list(A._profile[1])
                q_part = sum(1 << i for i, v in enumerate(k) if v == 2)
                if not truncated:
                    q_part |= (2 ** 32 - 1) & ~((1 << len(k)) - 1)
            fields = ['"truncated": %s' % ("true" if truncated else "false")]
            fields.append('"p_part": [%s]' % ", ".join(str(e) for e in p_part))
            if q_part is not None:
                fields.append('"q_part": %s' % q_part)
            algebra = ', "algebra": {"profile": {%s}}' % ", ".join(fields)
        return '{"p": %s%s, "type": "finite dimensional module", "gens": {%s}, "actions": [%s]}' % (
            p, algebra, gens, ", ".join(actions)
        )

    def check_associativity(self):
        """
        The triples `(a, b, x)` of Milnor basis elements and basis elements with
        `(a * b) * x != a * (b * x)` inside the module's range; empty if the action is associative.
        """
        A = self._A
        failures = []
        for key in self.basis().keys():
            x = self.monomial(key)
            room = self._t_max - self.degree_on_basis(key)
            for db in range(1, room + 1):
                for b in A.basis(db):
                    bx = self.act(b, x)
                    for da in range(1, room - db + 1):
                        for a in A.basis(da):
                            if self.act(a * b, x) != self.act(a, bx):
                                failures.append((a, b, x))
        return failures

    class Element(CombinatorialFreeModule.Element):
        """An element of a `SseqModule`, acted on by the module's Steenrod algebra."""

        def _acted_upon_(self, scalar, self_on_left):
            P = self.parent()
            if not self_on_left and scalar.parent() is P.algebra():
                return P.act(scalar, self)
            return CombinatorialFreeModule.Element._acted_upon_(self, scalar, self_on_left)


def _sseq_module(cls, p, keys, name):
    """A new module of class `cls` over `GF(p)` with basis `keys`, to be set up by the caller."""
    # The prefix keeps modules with the same basis keys apart, since Sage reuses equal parents.
    return cls(GF(p), keys, prefix=name, category=ModulesWithBasis(GF(p)).Graded())


def _sseq_prime_action(M):
    """
    Make the first product of `M.algebra()` with `M` one that cannot raise.

    Sage looks for an action on the first such product and remembers a failure.
    """
    M.algebra().one() * M.monomial(min(M.basis().keys()))


class SseqSteenrodModule(SseqModule):
    """
    A module known to sseq, given by the action of every Milnor basis element on every basis
    element through degree `t_max`. The basis keys are pairs `(t, i)`.
    """

    def setup(self, name, algebra, names, t_max, complete, actions):
        """Record the action table; use `sseq_steenrod_module` instead of calling this."""
        self._module_name = name
        self._A = algebra
        self._basis_names = names
        self._t_max = t_max
        self._complete = complete
        # (milnor_key, (t, i)) -> terms (coefficient, (t', j)); absent means zero.
        self._actions = actions
        self._truncation_hint = (
            "call the sseq MCP tool sseq_steenrod_module with module=%r and a larger t_max" % name
        )

    def _repr_term(self, key):
        return self._basis_names[key]

    def degree_on_basis(self, key):
        """The degree `t` of the basis element `(t, i)`."""
        return key[0]

    def _act_on_basis(self, m, key):
        return [(target, coeff) for coeff, target in self._actions.get((m, key), [])]


def sseq_steenrod_module(name, p, algebra, names, t_max, complete, actions):
    """A module known to sseq; the `sseq_steenrod_module` tool returns the call to make."""
    M = _sseq_module(SseqSteenrodModule, p, tuple(sorted(names)), name)
    M.setup(name, algebra, names, t_max, complete, actions)
    _sseq_prime_action(M)
    return M


class SseqTensorModule(SseqModule):
    """
    The tensor product `M # N` of two modules over the same Steenrod algebra, with the diagonal
    action `a (x # y) = sum (-1)^(|a''| |x|) a'x # a''y` from the coproduct of `a`.

    The basis keys are pairs of basis keys of `M` and `N`.
    """

    def setup(self, M, N, t_max, complete):
        """Record the factors; use `sseq_tensor` instead of calling this."""
        self._M = M
        self._N = N
        self._A = M.algebra()
        self._module_name = "%s # %s" % (M.module_name(), N.module_name())
        self._t_max = t_max
        self._complete = complete
        self._truncation_hint = "build the truncated factors with a larger t_max"

    def factors(self):
        """The factors `(M, N)`."""
        return (self._M, self._N)

    def _repr_term(self, key):
        M, N = self._M, self._N
        return "%s # %s" % (M._repr_term(key[0]), N._repr_term(key[1]))

    def degree_on_basis(self, key):
        """The sum of the degrees of the two factors."""
        return self._M.degree_on_basis(key[0]) + self._N.degree_on_basis(key[1])

    def _act_on_basis(self, m, key):
        M, N, A = self._M, self._N, self._A
        x, y = M.monomial(key[0]), N.monomial(key[1])
        odd = A.prime() != 2
        terms = []
        for (m1, m2), c in A.monomial(m).coproduct().monomial_coefficients().items():
            u = M.act(A.monomial(m1), x)
            if u == 0:
                continue
            w = N.act(A.monomial(m2), y)
            if odd and A.monomial(m2).degree() * M.degree_on_basis(key[0]) % 2:
                c = -c
            for j1, c1 in u:
                for j2, c2 in w:
                    terms.append(((j1, j2), c * c1 * c2))
        return terms


def sseq_tensor(M, N):
    """
    The tensor product `M # N` of two `SseqModule`s over the same algebra.

    If either factor is truncated, so is the product, at the largest degree in which all of its
    basis elements are known.
    """
    if M.algebra() is not N.algebra():
        raise ValueError(
            "the factors are modules over different algebras: %s and %s"
            % (M.algebra(), N.algebra())
        )
    # A truncated factor knows its basis up to its t_max, so the product knows degree d when every
    # basis element of the other factor leaves a degree within that range.
    bounds = []
    if not M.is_complete():
        bounds.append(M.t_max() + N.bottom())
    if not N.is_complete():
        bounds.append(N.t_max() + M.bottom())
    complete = not bounds
    t_max = min(bounds) if bounds else M.t_max() + N.t_max()
    keys = tuple(
        (k1, k2)
        for k1 in sorted(M.basis().keys())
        for k2 in sorted(N.basis().keys())
        if M.degree_on_basis(k1) + N.degree_on_basis(k2) <= t_max
    )
    name = "%s # %s" % (M.module_name(), N.module_name())
    T = _sseq_module(SseqTensorModule, M.algebra().prime(), keys, name)
    T.setup(M, N, t_max, complete)
    _sseq_prime_action(T)
    return T


def sseq_resolution(rid, p, algebra):
    """A resolution with no data yet; the `sseq_resolve` tool returns the call to make."""
    R = SseqResolution()
    R.setup(rid, p, algebra)
    return R
