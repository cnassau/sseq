//! One homological level of the resolution: a Gröbner basis of the graph of a differential.

use std::{
    hash::{BuildHasher, Hash},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::{
    ModMono, Mono, Params, Poly,
    pairs::{Pair, PairCounts, PairStore},
    poly::Merger,
};

/// The May filtration of a leading term together with its generator, ordered like Lin's `Filtr`:
/// larger generators are smaller.
pub type Filtration = (u32, u32);

/// The [`Filtration`] of `m`.
pub fn filtration(params: &Params, m: ModMono) -> Filtration {
    (!m.generator(), params.may_filtration(m.mono()))
}

/// An element $(x_1, x_2)$ of $C_s \oplus C_{s+1}$ with $d x_2 = x_1$.
///
/// When x2m pruning is on, `x2m` expresses the part of $x_1$ of filtration `fil` in the
/// associated graded of the May filtration, in terms of the leading parts of earlier elements. It
/// is only meaningful while `fil` is the filtration of the leading term of $x_1$, see
/// [`x2m_valid`](Self::x2m_valid).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GbElement {
    /// The component in $C_s$.
    pub x1: Poly,
    /// The component in $C_{s+1}$.
    pub x2: Poly,
    /// The May-graded shadow of $x_1$, on the x2m generators of the level.
    pub x2m: Poly,
    /// The filtration `x2m` belongs to.
    pub fil: Filtration,
}

impl Default for GbElement {
    fn default() -> Self {
        Self::new(Poly::zero(), Poly::zero())
    }
}

impl GbElement {
    /// The element $(x_1, x_2)$ without an x2m component.
    pub fn new(x1: Poly, x2: Poly) -> Self {
        Self {
            x1,
            x2,
            x2m: Poly::zero(),
            fil: (u32::MAX, u32::MAX),
        }
    }

    /// Whether `x2m` still describes the leading part of $x_1$. Once that part has cancelled, a
    /// non-zero `x2m` is a relation among the leading parts of earlier elements instead.
    pub fn x2m_valid(&self, params: &Params) -> bool {
        match self.x1.lead() {
            Some((lead, _)) => self.fil == filtration(params, lead),
            None => self.x2m.is_zero(),
        }
    }

    /// Multiply by the scalar `c`.
    pub fn scale(&mut self, c: u32, p: u32) {
        self.x1.scale(c, p);
        self.x2.scale(c, p);
        self.x2m.scale(c, p);
    }

    /// Scale so that the leading coefficient of `x1` is one.
    pub fn normalize(&mut self, params: &Params) {
        if let Some((_, c)) = self.x1.lead() {
            self.scale(params.inverse(c), params.prime());
        }
    }
}

/// Which component of a [`GbElement`] a cached product is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Component {
    X1,
    X2,
    X2m,
}

/// The x2m component a reduction carries along, and the filtration it belongs to.
pub struct X2mTrack<'a> {
    /// The x2m component.
    pub x2m: &'a mut Poly,
    /// The filtration of the element being reduced.
    pub fil: Filtration,
}

/// The [`ProductCache`]s of one degree stop storing products once they hold this many terms in
/// all. Products beyond it are computed again by every reduction that needs them, which trades
/// time for memory.
pub const CACHE_TERMS_BUDGET: usize = 1 << 23;

/// The number of independently locked parts of a [`ProductCache`]. More shards mean less
/// contention between threads, at the cost of a little memory.
const CACHE_SHARDS: usize = 16;

/// A key of a [`ProductCache`]: the element, the monomial and the component.
type ProductKey = (u32, Mono, Component);

/// The products $u \cdot g$ of the basis elements of one level by monomials that the reductions
/// of one internal degree need, shared between threads.
///
/// The same products recur across the reductions of a degree, and only there, since a product
/// serves to cancel a term of its own degree. Each product is computed once, by whichever thread
/// asks first; others asking meanwhile wait for it.
pub struct ProductCache {
    shards: Vec<Mutex<FxHashMap<ProductKey, Arc<OnceLock<Arc<Poly>>>>>>,
    terms: AtomicUsize,
}

impl Default for ProductCache {
    fn default() -> Self {
        Self {
            shards: (0..CACHE_SHARDS).map(|_| Mutex::default()).collect(),
            terms: AtomicUsize::new(0),
        }
    }
}

impl ProductCache {
    /// The product with the given key, computed by `compute` if it is not known yet. Nothing new
    /// is stored once `total`, the number of terms stored by all caches of the degree, reaches
    /// [`CACHE_TERMS_BUDGET`].
    fn get(
        &self,
        key: ProductKey,
        total: &AtomicUsize,
        compute: impl FnOnce() -> Poly,
    ) -> Arc<Poly> {
        let shard = FxBuildHasher.hash_one(key) as usize % CACHE_SHARDS;
        let cell = {
            let mut shard = self.shards[shard].lock().unwrap();
            match shard.get(&key) {
                Some(cell) => Arc::clone(cell),
                None if total.load(Ordering::Relaxed) >= CACHE_TERMS_BUDGET => {
                    drop(shard);
                    return Arc::new(compute());
                }
                None => Arc::clone(shard.entry(key).or_default()),
            }
        };
        Arc::clone(cell.get_or_init(|| {
            let prod = compute();
            self.terms.fetch_add(prod.num_terms(), Ordering::Relaxed);
            total.fetch_add(prod.num_terms(), Ordering::Relaxed);
            Arc::new(prod)
        }))
    }

    /// Drop everything, giving the terms back to `total`.
    pub fn clear(&self, total: &AtomicUsize) {
        for shard in &self.shards {
            *shard.lock().unwrap() = FxHashMap::default();
        }
        total.fetch_sub(self.terms.swap(0, Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// The cache of products of a [`Level`], and the budget it shares with the caches of the other
/// levels.
#[derive(Clone, Copy)]
pub struct CacheRef<'a> {
    /// The cache.
    pub cache: &'a ProductCache,
    /// The number of terms stored by all caches of the degree.
    pub total: &'a AtomicUsize,
}

/// A Gröbner basis of the graph of $d \colon C_{s+1} \to C_s$, ordered by the $C_s$ component.
///
/// Elements are normalized to have leading coefficient one.
///
/// The same structure holds the x2m bases, which are Gröbner bases of submodules of free modules
/// over the associated graded algebra of the May filtration. There only $x_1$ is used, and
/// products keep the terms of the right May filtration.
#[derive(Debug)]
pub struct Level {
    /// Whether products are taken in the associated graded algebra of the May filtration.
    may: bool,
    elements: Vec<GbElement>,
    leads: Vec<ModMono>,
    by_generator: FxHashMap<u32, Vec<u32>>,
    pairs: PairStore,
}

impl Level {
    /// An empty level whose pairs are truncated at `t_trunc`.
    pub fn new(t_trunc: i32) -> Self {
        Self::with_products(t_trunc, false)
    }

    /// An empty x2m basis whose pairs are truncated at `t_trunc`.
    pub fn new_may(t_trunc: i32) -> Self {
        Self::with_products(t_trunc, true)
    }

    /// An empty level with products in the associated graded algebra if `may` is set.
    fn with_products(t_trunc: i32, may: bool) -> Self {
        Self {
            may,
            elements: Vec::new(),
            leads: Vec::new(),
            by_generator: FxHashMap::default(),
            pairs: PairStore::new(t_trunc),
        }
    }

    /// The elements of the basis.
    pub fn elements(&self) -> &[GbElement] {
        &self.elements
    }

    /// The leading terms of the elements.
    pub fn leads(&self) -> &[ModMono] {
        &self.leads
    }

    /// The truncation degree of the critical pairs.
    pub fn t_trunc(&self) -> i32 {
        self.pairs.t_trunc()
    }

    /// Recompute the critical pairs for a new truncation degree, keeping those of degree at
    /// least `t_min`.
    pub fn retruncate(
        &mut self,
        params: &Params,
        generator_degree: impl Fn(u32) -> i32,
        t_trunc: i32,
        t_min: i32,
    ) {
        self.pairs = PairStore::rebuild(params, &self.leads, generator_degree, t_trunc, t_min);
    }

    /// Add an element with leading coefficient one, on generators of degrees given by
    /// `generator_degree`.
    pub fn push(&mut self, params: &Params, element: GbElement, generator_degree: i32) {
        let (lead, c) = element
            .x1
            .lead()
            .expect("pushing zero onto a Gröbner basis");
        debug_assert_eq!(c, 1);
        self.pairs
            .add_lead(params, &self.leads, lead, generator_degree);
        self.by_generator
            .entry(lead.generator())
            .or_default()
            .push(self.elements.len() as u32);
        self.leads.push(lead);
        self.elements.push(element);
    }

    /// Take the critical pairs of degree `t`.
    pub fn take_pairs(&mut self, params: &Params, t: i32) -> (Vec<Pair>, PairCounts) {
        self.pairs.take(params, &self.leads, t)
    }

    /// The index of an element whose leading term divides `m`.
    pub fn find_divisor(&self, params: &Params, m: ModMono) -> Option<usize> {
        self.by_generator
            .get(&m.generator())?
            .iter()
            .map(|&k| k as usize)
            .find(|&k| params.divides(self.leads[k].mono(), m.mono()))
    }

    /// The product `m * x`, in the associated graded algebra for an x2m basis.
    fn mul(&self, params: &Params, m: Mono, x: &Poly) -> Poly {
        if self.may {
            params.mul_may_poly(m, 1, x)
        } else {
            params.mul_poly(m, 1, x)
        }
    }

    /// The element of a critical pair or single, with the leading terms cancelled. With
    /// `track_x2m`, its x2m component and filtration are computed as well.
    pub fn pair_element(&self, params: &Params, pair: &Pair, track_x2m: bool) -> GbElement {
        let p = params.prime();
        let side = |k: u32, m: Mono| {
            let g = &self.elements[k as usize];
            let mut e = GbElement::new(self.mul(params, m, &g.x1), self.mul(params, m, &g.x2));
            if track_x2m {
                e.x2m = params.mul_may_poly(m, 1, &g.x2m);
                e.fil = (g.fil.0, g.fil.1 + params.may_filtration(m));
            }
            e
        };
        let mut b = side(pair.i2, pair.m2);
        if pair.is_single() {
            return b;
        }
        let a = side(pair.i1, pair.m1);
        let (la, ca) = a.x1.lead().unwrap();
        let (lb, cb) = b.x1.lead().unwrap();
        debug_assert_eq!(la, lb, "the leading terms of a critical pair differ");
        // ca * b - cb * a
        b.scale(ca, p);
        let c = params.neg(cb);
        b.x1.add_scaled(&a.x1, c, p);
        b.x2.add_scaled(&a.x2, c, p);
        b.x2m.add_scaled(&a.x2m, c, p);
        b
    }

    /// The x2m component of a critical pair or single, as in [`pair_element`](Self::pair_element).
    pub fn pair_x2m(&self, params: &Params, pair: &Pair) -> Poly {
        let p = params.prime();
        let g2 = &self.elements[pair.i2 as usize];
        let mut x = params.mul_may_poly(pair.m2, 1, &g2.x2m);
        if pair.is_single() {
            return x;
        }
        let g1 = &self.elements[pair.i1 as usize];
        let ca = params.lf_coefficient(pair.m1, self.leads[pair.i1 as usize].mono());
        let cb = params.lf_coefficient(pair.m2, self.leads[pair.i2 as usize].mono());
        x.scale(ca, p);
        x.add_scaled(&params.mul_may_poly(pair.m1, 1, &g1.x2m), params.neg(cb), p);
        x
    }

    /// The filtration of the x2m component of a critical pair or single.
    pub fn pair_filtration(&self, params: &Params, pair: &Pair) -> Filtration {
        let fil = self.elements[pair.i2 as usize].fil;
        (fil.0, fil.1 + params.may_filtration(pair.m2))
    }

    /// The product of the monomial `u` with a component of element `k`.
    fn product(
        &self,
        params: &Params,
        cache: CacheRef,
        k: usize,
        u: Mono,
        component: Component,
    ) -> Arc<Poly> {
        cache.cache.get((k as u32, u, component), cache.total, || {
            let g = &self.elements[k];
            match component {
                Component::X1 => self.mul(params, u, &g.x1),
                Component::X2 => self.mul(params, u, &g.x2),
                Component::X2m => params.mul_may_poly(u, 1, &g.x2m),
            }
        })
    }

    /// Reduce `x` completely modulo the $C_s$ components, applying the same operations to
    /// `companion`, if given. Returns the number of reduction steps.
    ///
    /// With `track`, a step that cancels a term of the tracked filtration while the leading term
    /// of `x` still has that filtration also applies to the x2m component, in the associated
    /// graded algebra.
    pub fn reduce(
        &self,
        params: &Params,
        cache: CacheRef,
        x: &mut Poly,
        companion: Option<&mut Poly>,
        track: Option<X2mTrack>,
    ) -> u64 {
        let p = params.prime();
        let mut merger = Merger::default();
        merger.add(Arc::new(std::mem::take(x)), 1, 0);
        let mut companions: Vec<(u32, Arc<Poly>)> = Vec::new();
        let mut x2ms: Vec<(u32, Arc<Poly>)> = Vec::new();
        let mut out: Vec<(ModMono, u32)> = Vec::new();
        let mut steps = 0;
        while let Some((m, c)) = merger.next_term(p) {
            let Some(k) = self.find_divisor(params, m) else {
                out.push((m, c));
                continue;
            };
            steps += 1;
            let u = params.quotient(m.mono(), self.leads[k].mono());
            let prod = self.product(params, cache, k, u, Component::X1);
            let (lead, lambda) = prod.lead().unwrap();
            debug_assert_eq!(lead, m);
            let factor = params.neg(c * params.inverse(lambda) % p);
            if let Some(track) = &track {
                let current_lead = out.first().map_or(m, |&(l, _)| l);
                if filtration(params, current_lead) == track.fil
                    && filtration(params, m) == track.fil
                {
                    x2ms.push((factor, self.product(params, cache, k, u, Component::X2m)));
                }
            }
            // The leading term of the product cancels `m`.
            merger.add(prod, factor, 1);
            if companion.is_some() {
                companions.push((factor, self.product(params, cache, k, u, Component::X2)));
            }
        }
        *x = Poly::from_sorted_terms(out);
        let merge_into = |target: &mut Poly, parts: Vec<(u32, Arc<Poly>)>| {
            if parts.is_empty() {
                return;
            }
            let mut merger = Merger::default();
            merger.add(Arc::new(std::mem::take(target)), 1, 0);
            for (factor, prod) in parts {
                merger.add(prod, factor, 0);
            }
            *target = merger.finish(p);
        };
        if let Some(companion) = companion {
            merge_into(companion, companions);
        }
        if let Some(track) = track {
            merge_into(track.x2m, x2ms);
        }
        steps
    }

    /// Reduce the leading terms of `r` against `against`, which must all be of the same degree as
    /// `r` and triangular among themselves. With `track_x2m`, the x2m components are added
    /// whenever the filtrations agree and that of `r` is valid.
    pub fn triangulate_element(
        params: &Params,
        r: &mut GbElement,
        against: &[GbElement],
        track_x2m: bool,
    ) {
        let p = params.prime();
        for g in against {
            let (lead, _) = g.x1.lead().unwrap();
            let c = r.x1.coefficient(lead);
            if c == 0 {
                continue;
            }
            let c = params.neg(c);
            if track_x2m && r.fil == g.fil && r.x2m_valid(params) {
                r.x2m.add_scaled(&g.x2m, c, p);
            }
            r.x1.add_scaled(&g.x1, c, p);
            r.x2.add_scaled(&g.x2, c, p);
        }
    }

    /// Reduce the leading terms of `x` against the elements with the given indices, which must
    /// all be of the same degree as `x` and triangular among themselves.
    pub fn triangulate(
        params: &Params,
        x: &mut Poly,
        mut companion: Option<&mut Poly>,
        against: &[GbElement],
    ) {
        let p = params.prime();
        for g in against {
            let (lead, _) = g.x1.lead().unwrap();
            let c = x.coefficient(lead);
            if c != 0 {
                x.add_scaled(&g.x1, params.neg(c), p);
                if let Some(companion) = companion.as_deref_mut() {
                    companion.add_scaled(&g.x2, params.neg(c), p);
                }
            }
        }
    }
}
