//! One homological level of the resolution: a Gröbner basis of the graph of a differential.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use rustc_hash::FxHashMap;

use crate::{
    ModMono, Mono, Params, Poly,
    pairs::{Pair, PairCounts, PairStore},
    poly::Merger,
};

/// An element $(x_1, x_2)$ of $C_s \oplus C_{s+1}$ with $d x_2 = x_1$.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GbElement {
    /// The component in $C_s$.
    pub x1: Poly,
    /// The component in $C_{s+1}$.
    pub x2: Poly,
}

impl GbElement {
    /// Add `c * other`.
    pub fn add_scaled(&mut self, other: &Self, c: u32, p: u32) {
        self.x1.add_scaled(&other.x1, c, p);
        self.x2.add_scaled(&other.x2, c, p);
    }

    /// Multiply by the scalar `c`.
    pub fn scale(&mut self, c: u32, p: u32) {
        self.x1.scale(c, p);
        self.x2.scale(c, p);
    }

    /// Scale so that the leading coefficient of `x1` is one.
    pub fn normalize(&mut self, params: &Params) {
        if let Some((_, c)) = self.x1.lead() {
            self.scale(params.inverse(c), params.prime());
        }
    }
}

/// A product cache that grows beyond this many terms is cleared, trading recomputation for
/// memory.
const CACHE_TERMS_LIMIT: usize = 1 << 20;

/// Which component of a [`GbElement`] a cached product is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Component {
    X1,
    X2,
}

/// Products $u \cdot g$ of basis elements by monomials already computed by this thread, valid
/// for one [`Params`] and one cache epoch.
///
/// Elements of a level never change once pushed, so a product stays correct as long as the
/// level exists. The cache is cleared with the product cache of `Params`, i.e. in every internal
/// degree, since a product only serves to reduce terms of its own degree.
#[derive(Default)]
struct ElementProductCache {
    owner: (u64, u64),
    products: FxHashMap<(u64, u32, Mono, Component), Rc<Poly>>,
    terms: usize,
}

thread_local! {
    static ELEMENT_PRODUCTS: RefCell<ElementProductCache> =
        RefCell::new(ElementProductCache::default());
}

/// A Gröbner basis of the graph of $d \colon C_{s+1} \to C_s$, ordered by the $C_s$ component.
///
/// Elements are normalized to have leading coefficient one.
#[derive(Debug)]
pub struct Level {
    /// Distinguishes levels in the thread-local product caches.
    id: u64,
    elements: Vec<GbElement>,
    leads: Vec<ModMono>,
    by_generator: FxHashMap<u32, Vec<u32>>,
    pairs: PairStore,
}

impl Level {
    /// An empty level whose pairs are truncated at `t_trunc`.
    pub fn new(t_trunc: i32) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
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
            .entry(lead.generator)
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
            .get(&m.generator)?
            .iter()
            .map(|&k| k as usize)
            .find(|&k| params.divides(self.leads[k].mono, m.mono))
    }

    /// The element of a critical pair or single, with the leading terms cancelled.
    pub fn pair_element(&self, params: &Params, pair: &Pair) -> GbElement {
        let p = params.prime();
        let g2 = &self.elements[pair.i2 as usize];
        let mut b = GbElement {
            x1: params.mul_poly(pair.m2, 1, &g2.x1),
            x2: params.mul_poly(pair.m2, 1, &g2.x2),
        };
        if pair.is_single() {
            return b;
        }
        let g1 = &self.elements[pair.i1 as usize];
        let a = GbElement {
            x1: params.mul_poly(pair.m1, 1, &g1.x1),
            x2: params.mul_poly(pair.m1, 1, &g1.x2),
        };
        let (la, ca) = a.x1.lead().unwrap();
        let (lb, cb) = b.x1.lead().unwrap();
        debug_assert_eq!(la, lb, "the leading terms of a critical pair differ");
        // ca * b - cb * a
        b.scale(ca, p);
        b.add_scaled(&a, params.neg(cb), p);
        b
    }

    /// The product of the monomial `u` with a component of element `k`.
    fn product(&self, params: &Params, k: usize, u: Mono, component: Component) -> Rc<Poly> {
        ELEMENT_PRODUCTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            let owner = (params.id(), params.cache_epoch());
            if cache.owner != owner || cache.terms > CACHE_TERMS_LIMIT {
                cache.owner = owner;
                cache.products.clear();
                cache.terms = 0;
            }
            let key = (self.id, k as u32, u, component);
            if let Some(prod) = cache.products.get(&key) {
                return Rc::clone(prod);
            }
            let g = &self.elements[k];
            let x = match component {
                Component::X1 => &g.x1,
                Component::X2 => &g.x2,
            };
            let prod = Rc::new(params.mul_poly(u, 1, x));
            cache.terms += prod.num_terms();
            cache.products.insert(key, Rc::clone(&prod));
            prod
        })
    }

    /// Reduce `x` completely modulo the $C_s$ components, applying the same operations to
    /// `companion`, if given. Returns the number of reduction steps.
    pub fn reduce(&self, params: &Params, x: &mut Poly, companion: Option<&mut Poly>) -> u64 {
        let p = params.prime();
        let mut merger = Merger::default();
        merger.add(Rc::new(std::mem::take(x)), 1, 0);
        let mut companions: Vec<(u32, Rc<Poly>)> = Vec::new();
        let mut out = Vec::new();
        let mut steps = 0;
        while let Some((m, c)) = merger.next_term(p) {
            let Some(k) = self.find_divisor(params, m) else {
                out.push((m, c));
                continue;
            };
            steps += 1;
            let u = params.quotient(m.mono, self.leads[k].mono);
            let prod = self.product(params, k, u, Component::X1);
            let (lead, lambda) = prod.lead().unwrap();
            debug_assert_eq!(lead, m);
            let factor = params.neg(c * params.inverse(lambda) % p);
            // The leading term of the product cancels `m`.
            merger.add(prod, factor, 1);
            if companion.is_some() {
                companions.push((factor, self.product(params, k, u, Component::X2)));
            }
        }
        *x = Poly::from_sorted_terms(out);
        if let Some(companion) = companion
            && !companions.is_empty()
        {
            let mut merger = Merger::default();
            merger.add(Rc::new(std::mem::take(companion)), 1, 0);
            for (factor, prod) in companions {
                merger.add(prod, factor, 0);
            }
            *companion = merger.finish(p);
        }
        steps
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
