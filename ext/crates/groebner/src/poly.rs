//! Sparse elements of free modules over the Steenrod algebra.

use std::{cmp::Reverse, collections::BinaryHeap, sync::Arc};

use crate::{ModMono, Mono, Params};

/// An element of a free module, as terms sorted in increasing order with non-zero coefficients.
///
/// The monomials and the coefficients are kept apart, so that a term takes 17 bytes rather than
/// the 32 of an aligned pair. Coefficients are residues mod a prime of at most [`MAX_PRIME`].
///
/// The leading term is the first one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    monos: Vec<ModMono>,
    coeffs: Vec<u8>,
}

/// The largest prime whose residues fit the coefficients of a [`Poly`].
pub const MAX_PRIME: u32 = u8::MAX as u32;

impl Poly {
    /// The zero element.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The element `c * m`.
    pub fn monomial(m: ModMono, c: u32) -> Self {
        Self {
            monos: vec![m],
            coeffs: vec![c as u8],
        }
    }

    /// An empty element with room for `n` terms.
    fn with_capacity(n: usize) -> Self {
        Self {
            monos: Vec::with_capacity(n),
            coeffs: Vec::with_capacity(n),
        }
    }

    /// Append a term larger than all others, if its coefficient is non-zero.
    fn push(&mut self, m: ModMono, c: u32) {
        debug_assert!(self.monos.last().is_none_or(|&l| l < m));
        if c != 0 {
            self.monos.push(m);
            self.coeffs.push(c as u8);
        }
    }

    /// The sum of the given terms, which may be unsorted and repeat monomials.
    pub fn from_terms(mut terms: Vec<(ModMono, u32)>, p: u32) -> Self {
        terms.sort_unstable_by_key(|&(m, _)| m);
        let mut out = Self::with_capacity(terms.len());
        let mut pending: Option<(ModMono, u32)> = None;
        for (m, c) in terms {
            match &mut pending {
                Some((last, d)) if *last == m => *d = (*d + c) % p,
                _ => {
                    if let Some((last, d)) = pending {
                        out.push(last, d);
                    }
                    pending = Some((m, c % p));
                }
            }
        }
        if let Some((last, d)) = pending {
            out.push(last, d);
        }
        out.monos.shrink_to_fit();
        out.coeffs.shrink_to_fit();
        out
    }

    /// The element with the given terms, which must be sorted, distinct and non-zero.
    pub fn from_sorted_terms(terms: impl IntoIterator<Item = (ModMono, u32)>) -> Self {
        let mut out = Self::zero();
        for (m, c) in terms {
            debug_assert!(c != 0);
            out.push(m, c);
        }
        out
    }

    /// The terms in increasing order.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (ModMono, u32)> + '_ {
        self.monos
            .iter()
            .zip(&self.coeffs)
            .map(|(&m, &c)| (m, c as u32))
    }

    /// The term in position `i`.
    pub fn term(&self, i: usize) -> (ModMono, u32) {
        (self.monos[i], self.coeffs[i] as u32)
    }

    /// The number of terms.
    pub fn num_terms(&self) -> usize {
        self.monos.len()
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.monos.is_empty()
    }

    /// The leading term.
    pub fn lead(&self) -> Option<(ModMono, u32)> {
        (!self.is_zero()).then(|| self.term(0))
    }

    /// The coefficient of `m`.
    pub fn coefficient(&self, m: ModMono) -> u32 {
        self.monos
            .binary_search(&m)
            .map_or(0, |i| self.coeffs[i] as u32)
    }

    /// Multiply by the scalar `c`.
    pub fn scale(&mut self, c: u32, p: u32) {
        if c.is_multiple_of(p) {
            *self = Self::zero();
            return;
        }
        for d in &mut self.coeffs {
            *d = (*d as u32 * c % p) as u8;
        }
    }

    /// Add `c * other` to `self`.
    pub fn add_scaled(&mut self, other: &Self, c: u32, p: u32) {
        if c.is_multiple_of(p) || other.is_zero() {
            return;
        }
        let mut out = Self::with_capacity(self.num_terms() + other.num_terms());
        let (mut i, mut j) = (0, 0);
        while i < self.num_terms() && j < other.num_terms() {
            let (ma, ca) = self.term(i);
            let (mb, cb) = other.term(j);
            match ma.cmp(&mb) {
                std::cmp::Ordering::Less => {
                    out.push(ma, ca);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(mb, cb * c % p);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push(ma, (ca + cb * c) % p);
                    i += 1;
                    j += 1;
                }
            }
        }
        for k in i..self.num_terms() {
            let (m, d) = self.term(k);
            out.push(m, d);
        }
        for k in j..other.num_terms() {
            let (m, d) = other.term(k);
            out.push(m, d * c % p);
        }
        *self = out;
    }
}

/// A linear combination of polynomials, read off one term at a time in increasing order.
///
/// Summands can be added while reading, as long as their terms are not smaller than the last
/// term read. This sums $k$ polynomials with $n$ terms in all in time $O(n \log k)$, which is what
/// makes reduction cheap: the element being reduced is never copied.
#[derive(Default)]
pub struct Merger {
    sources: Vec<(Arc<Poly>, u32, usize)>,
    heap: BinaryHeap<Reverse<(ModMono, u32)>>,
}

impl Merger {
    /// Add `factor` times the terms of `poly` from position `start` on.
    pub fn add(&mut self, poly: Arc<Poly>, factor: u32, start: usize) {
        if factor == 0 || start >= poly.num_terms() {
            return;
        }
        let idx = self.sources.len() as u32;
        self.heap.push(Reverse((poly.monos[start], idx)));
        self.sources.push((poly, factor, start));
    }

    /// Step past the current term of summand `idx`, returning its scaled coefficient, not yet
    /// reduced mod p.
    fn advance(&mut self, idx: u32) -> u32 {
        let (poly, factor, pos) = &mut self.sources[idx as usize];
        let c = poly.coeffs[*pos] as u32 * *factor;
        *pos += 1;
        if let Some(&m) = poly.monos.get(*pos) {
            self.heap.push(Reverse((m, idx)));
        }
        c
    }

    /// The next term with a non-zero coefficient.
    pub fn next_term(&mut self, p: u32) -> Option<(ModMono, u32)> {
        while let Some(Reverse((m, idx))) = self.heap.pop() {
            let mut c = self.advance(idx) % p;
            while let Some(&Reverse((m2, idx2))) = self.heap.peek() {
                if m2 != m {
                    break;
                }
                self.heap.pop();
                c = (c + self.advance(idx2)) % p;
            }
            if c != 0 {
                return Some((m, c));
            }
        }
        None
    }

    /// The sum of everything not read yet.
    pub fn finish(mut self, p: u32) -> Poly {
        let mut out = Poly::zero();
        while let Some((m, c)) = self.next_term(p) {
            out.push(m, c);
        }
        out
    }
}

impl Params {
    /// The product `c * m * x`.
    pub fn mul_poly(&self, m: Mono, c: u32, x: &Poly) -> Poly {
        let p = self.prime();
        let mut terms = Vec::with_capacity(x.num_terms());
        for (t, d) in x.terms() {
            let cd = c * d % p;
            self.mul(m, t.mono(), |prod, e| {
                terms.push((ModMono::new(t.generator(), prod), cd * e % p))
            });
        }
        Poly::from_terms(terms, p)
    }

    /// The product `c * m * x` in the associated graded algebra of the May filtration: the terms
    /// of the product whose May filtration is the sum of those of the factors.
    pub fn mul_may_poly(&self, m: Mono, c: u32, x: &Poly) -> Poly {
        let p = self.prime();
        let wm = self.may_filtration(m);
        let mut terms = Vec::with_capacity(x.num_terms());
        for (t, d) in x.terms() {
            let cd = c * d % p;
            let w = wm + self.may_filtration(t.mono());
            self.mul(m, t.mono(), |prod, e| {
                if self.may_filtration(prod) == w {
                    terms.push((ModMono::new(t.generator(), prod), cd * e % p));
                }
            });
        }
        Poly::from_terms(terms, p)
    }

    /// Format a module element, naming generator `g` by `name(g)`.
    pub fn format_poly(&self, x: &Poly, name: impl Fn(u32) -> String) -> String {
        if x.is_zero() {
            return "0".to_string();
        }
        let terms: Vec<String> = x
            .terms()
            .map(|(m, c)| {
                let coef = if c == 1 {
                    String::new()
                } else {
                    format!("{c} ")
                };
                let mono = if m.mono().is_one() {
                    String::new()
                } else {
                    format!("{} ", self.format(m.mono()))
                };
                format!("{coef}{mono}{}", name(m.generator()))
            })
            .collect();
        terms.join(" + ")
    }
}
