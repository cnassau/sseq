//! Sparse elements of free modules over the Steenrod algebra.

use std::{cmp::Reverse, collections::BinaryHeap, rc::Rc};

use crate::{ModMono, Mono, Params};

/// An element of a free module, as terms sorted in increasing order with non-zero coefficients.
///
/// The leading term is the first one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    terms: Vec<(ModMono, u32)>,
}

impl Poly {
    /// The zero element.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The element `c * m`.
    pub fn monomial(m: ModMono, c: u32) -> Self {
        Self {
            terms: vec![(m, c)],
        }
    }

    /// The sum of the given terms, which may be unsorted and repeat monomials.
    pub fn from_terms(mut terms: Vec<(ModMono, u32)>, p: u32) -> Self {
        terms.sort_unstable_by_key(|&(m, _)| m);
        let mut out: Vec<(ModMono, u32)> = Vec::with_capacity(terms.len());
        for (m, c) in terms {
            match out.last_mut() {
                Some((last, d)) if *last == m => *d = (*d + c) % p,
                _ => {
                    if let Some((_, 0)) = out.last() {
                        out.pop();
                    }
                    out.push((m, c % p));
                }
            }
        }
        if let Some((_, 0)) = out.last() {
            out.pop();
        }
        Self { terms: out }
    }

    /// The element with the given terms, which must be sorted, distinct and non-zero.
    pub fn from_sorted_terms(terms: Vec<(ModMono, u32)>) -> Self {
        debug_assert!(terms.is_sorted_by(|a, b| a.0 < b.0));
        debug_assert!(terms.iter().all(|&(_, c)| c != 0));
        Self { terms }
    }

    /// The terms in increasing order.
    pub fn terms(&self) -> &[(ModMono, u32)] {
        &self.terms
    }

    /// The number of terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The leading term.
    pub fn lead(&self) -> Option<(ModMono, u32)> {
        self.terms.first().copied()
    }

    /// The coefficient of `m`.
    pub fn coefficient(&self, m: ModMono) -> u32 {
        self.terms
            .binary_search_by(|&(x, _)| x.cmp(&m))
            .map_or(0, |i| self.terms[i].1)
    }

    /// Multiply by the scalar `c`.
    pub fn scale(&mut self, c: u32, p: u32) {
        if c.is_multiple_of(p) {
            self.terms.clear();
        }
        for (_, d) in &mut self.terms {
            *d = *d * c % p;
        }
    }

    /// Add `c * other` to `self`.
    pub fn add_scaled(&mut self, other: &Self, c: u32, p: u32) {
        if c.is_multiple_of(p) || other.is_zero() {
            return;
        }
        let mut out = Vec::with_capacity(self.terms.len() + other.terms.len());
        let (mut a, mut b) = (self.terms.iter().peekable(), other.terms.iter().peekable());
        loop {
            match (a.peek(), b.peek()) {
                (Some(&&(ma, ca)), Some(&&(mb, cb))) => match ma.cmp(&mb) {
                    std::cmp::Ordering::Less => {
                        out.push((ma, ca));
                        a.next();
                    }
                    std::cmp::Ordering::Greater => {
                        out.push((mb, cb * c % p));
                        b.next();
                    }
                    std::cmp::Ordering::Equal => {
                        let s = (ca + cb * c) % p;
                        if s != 0 {
                            out.push((ma, s));
                        }
                        a.next();
                        b.next();
                    }
                },
                (Some(&&t), None) => {
                    out.push(t);
                    a.next();
                }
                (None, Some(&&(mb, cb))) => {
                    out.push((mb, cb * c % p));
                    b.next();
                }
                (None, None) => break,
            }
        }
        self.terms = out;
    }
}

/// A linear combination of polynomials, read off one term at a time in increasing order.
///
/// Summands can be added while reading, as long as their terms are not smaller than the last
/// term read. This sums $k$ polynomials with $n$ terms in all in time $O(n \log k)$, which is what
/// makes reduction cheap: the element being reduced is never copied.
#[derive(Default)]
pub struct Merger {
    sources: Vec<(Rc<Poly>, u32, usize)>,
    heap: BinaryHeap<Reverse<(ModMono, u32)>>,
}

impl Merger {
    /// Add `factor` times the terms of `poly` from position `start` on.
    pub fn add(&mut self, poly: Rc<Poly>, factor: u32, start: usize) {
        if factor == 0 || start >= poly.num_terms() {
            return;
        }
        let idx = self.sources.len() as u32;
        self.heap.push(Reverse((poly.terms[start].0, idx)));
        self.sources.push((poly, factor, start));
    }

    /// Step past the current term of summand `idx`, returning its scaled coefficient, not yet
    /// reduced mod p.
    fn advance(&mut self, idx: u32) -> u32 {
        let (poly, factor, pos) = &mut self.sources[idx as usize];
        let c = poly.terms[*pos].1 * *factor;
        *pos += 1;
        if let Some(&(m, _)) = poly.terms.get(*pos) {
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
        let mut terms = Vec::new();
        while let Some(t) = self.next_term(p) {
            terms.push(t);
        }
        Poly { terms }
    }
}

impl Params {
    /// The product `c * m * x`.
    pub fn mul_poly(&self, m: Mono, c: u32, x: &Poly) -> Poly {
        let p = self.prime();
        let mut terms = Vec::with_capacity(x.num_terms());
        for &(t, d) in x.terms() {
            let cd = c * d % p;
            self.mul_cached(m, t.mono, |prod, e| {
                terms.push((ModMono::new(t.generator, prod), cd * e % p))
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
            .iter()
            .map(|&(m, c)| {
                let coef = if c == 1 {
                    String::new()
                } else {
                    format!("{c} ")
                };
                let mono = if m.mono.is_one() {
                    String::new()
                } else {
                    format!("{} ", self.format(m.mono))
                };
                format!("{coef}{mono}{}", name(m.generator))
            })
            .collect();
        terms.join(" + ")
    }
}
