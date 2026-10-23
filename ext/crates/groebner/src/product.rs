//! The Milnor product formula on packed monomials.

use crate::{Mono, Params};

/// The longest sequence $R$ in $P(R)$, which bounds the index $n$ of $\xi_n$.
pub(crate) const MAX_LEN: usize = 16;

/// A sequence $R = (r_1, r_2, \dots)$ on the stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seq {
    len: usize,
    entries: [u32; MAX_LEN],
}

impl Seq {
    /// The entries $r_1, \dots, r_{len}$.
    fn as_slice(&self) -> &[u32] {
        &self.entries[..self.len]
    }

    /// Drop trailing zeros.
    fn trim(&mut self) {
        while self.len > 0 && self.entries[self.len - 1] == 0 {
            self.len -= 1;
        }
    }
}

impl Params {
    /// The exterior part and the sequence $R$ of `m`.
    pub(crate) fn decode_seq(&self, m: Mono) -> (u32, Seq) {
        let mut r = Seq::default();
        if let Some(bits) = self.bit_positions() {
            let mut rest = m.digits();
            while rest != 0 {
                let (n, value) = bits[rest.trailing_zeros() as usize];
                rest &= rest - 1;
                r.entries[n as usize] += value;
                r.len = r.len.max(n as usize + 1);
            }
        } else {
            for (idx, d) in self.digits(m) {
                let pos = &self.positions()[idx];
                let n = pos.xi as usize;
                r.len = r.len.max(n);
                r.entries[n - 1] += d * pos.value;
            }
        }
        (m.q_part(), r)
    }

    /// Call `f(m, c)` for every term $c \cdot m$ of the product `a * b`.
    ///
    /// A monomial may be reported more than once, and the coefficients are non-zero.
    pub fn mul(&self, a: Mono, b: Mono, mut f: impl FnMut(Mono, u32)) {
        if a.is_one() {
            return f(b, 1);
        }
        if b.is_one() {
            return f(a, 1);
        }
        let (ea, r) = self.decode_seq(a);
        let (eb, s) = self.decode_seq(b);
        if eb == 0 {
            self.mul_ppart(&r, &s, |t, c| f(self.encode_product(ea, t), c));
            return;
        }
        for (c0, e, r) in self.move_q_left(ea, r, eb) {
            self.mul_ppart(&r, &s, |t, c| {
                f(self.encode_product(e, t), c0 * c % self.prime())
            });
        }
    }

    /// The monomial $Q(E) P(T)$ of a product, which is representable whenever the factors are of
    /// degree at most [`max_degree`](Self::max_degree) in total.
    fn encode_product(&self, q_part: u32, t: &[u32]) -> Mono {
        self.encode(q_part, t).unwrap_or_else(|| {
            panic!(
                "product Q({q_part:b})P({t:?}) exceeds the supported degree {}",
                self.max_degree()
            )
        })
    }

    /// Expand $Q(E) P(R) Q(F)$ as a sum of $c \cdot Q(E') P(R')$, using
    /// $P(R) Q_k = Q_k P(R) + \sum_{i \geq 1} Q_{k+i} P(R - p^k e_i)$ and $Q_j Q_k = -Q_k Q_j$.
    fn move_q_left(&self, e: u32, r: Seq, f: u32) -> Vec<(u32, u32, Seq)> {
        let p = self.prime();
        let mut result = vec![(1, e, r)];
        let mut old = Vec::new();
        for k in (0..u32::BITS).filter(|k| f & (1 << k) != 0) {
            let pk = p.pow(k);
            std::mem::swap(&mut result, &mut old);
            result.clear();
            for (coef, e, r) in old.drain(..) {
                for i in 0..=r.len {
                    let qi = k + i as u32;
                    if e & (1 << qi) != 0 {
                        continue;
                    }
                    let mut new_r = r;
                    if i > 0 {
                        if new_r.entries[i - 1] < pk {
                            continue;
                        }
                        new_r.entries[i - 1] -= pk;
                        new_r.trim();
                    }
                    let larger = (e >> (qi + 1)).count_ones();
                    let c = if larger.is_multiple_of(2) {
                        coef
                    } else {
                        self.neg(coef)
                    };
                    result.push((c, e | (1 << qi), new_r));
                }
            }
        }
        result
    }

    /// Call `f(T, c)` for every term $c \cdot P(T)$ of $P(R) P(S)$ with $c \neq 0$, by
    /// enumerating Milnor matrices. Terms may repeat.
    pub(crate) fn mul_ppart(&self, r: &Seq, s: &Seq, mut f: impl FnMut(&[u32], u32)) {
        if r.len == 0 {
            return f(s.as_slice(), 1);
        }
        if s.len == 0 {
            return f(r.as_slice(), 1);
        }
        if self.prime() == 2 {
            return antidiagonal_terms(r, s, |t| f(t, 1));
        }
        let mut enumerator = MatrixEnumerator {
            params: self,
            p: self.prime(),
            r: r.as_slice(),
            cols: s.len + 1,
            col_rem: [0; MAX_LEN + 1],
            diagonal: [0; 2 * MAX_LEN + 1],
        };
        enumerator.col_rem[1..=s.len].copy_from_slice(s.as_slice());
        enumerator.row(1, 1, &mut f);
    }

    /// The binomial coefficient $\binom{a + b}{b}$ mod p, by Lucas' theorem.
    fn binomial_sum(&self, a: u32, b: u32) -> u32 {
        self.multinomial(&mut [a, b])
    }
}

/// The state of the enumeration of Milnor matrices $X$ with $\sum_j p^j x_{ij} = r_i$ and
/// $\sum_i x_{ij} = s_j$, at odd primes.
///
/// Cells are filled row by row. The coefficient of a matrix is the product over the diagonals of
/// the multinomial coefficients of their entries, which is accumulated cell by cell so that a
/// partial matrix with a vanishing coefficient is abandoned at once.
struct MatrixEnumerator<'a> {
    params: &'a Params,
    p: u32,
    r: &'a [u32],
    cols: usize,
    /// `col_rem[j]` is what is left of $s_j$ for the rows not yet filled, eventually $x_{0j}$.
    col_rem: [u32; MAX_LEN + 1],
    /// `diagonal[n]` is the sum of the entries placed so far on the diagonal $i + j = n$.
    diagonal: [u32; 2 * MAX_LEN + 1],
}

impl MatrixEnumerator<'_> {
    /// Place `v` on diagonal `n`, returning the new coefficient, or zero.
    fn place(&self, n: usize, v: u32, coef: u32) -> u32 {
        let c = self.params.binomial_sum(self.diagonal[n], v);
        coef * c % self.p
    }

    /// Fill row `i` and everything below it.
    fn row(&mut self, i: usize, coef: u32, f: &mut impl FnMut(&[u32], u32)) {
        if i > self.r.len() {
            return self.finish(coef, f);
        }
        self.cell(i, 1, self.r[i - 1], 1, coef, f);
    }

    /// Fill cell `(i, j)` and everything after it, with `budget` left of $r_i$ and `pj` the weight
    /// $p^{j-1}$ of the previous column.
    fn cell(
        &mut self,
        i: usize,
        j: usize,
        budget: u32,
        pj: u32,
        coef: u32,
        f: &mut impl FnMut(&[u32], u32),
    ) {
        if j == self.cols {
            // x_{i0} takes what is left of r_i.
            let c = self.place(i, budget, coef);
            if c != 0 {
                self.diagonal[i] += budget;
                self.row(i + 1, c, f);
                self.diagonal[i] -= budget;
            }
            return;
        }
        let pj = pj.saturating_mul(self.p);
        let max = (budget / pj).min(self.col_rem[j]);
        let n = i + j;
        let mut v = 0;
        loop {
            let c = self.place(n, v, coef);
            if c != 0 {
                self.diagonal[n] += v;
                self.col_rem[j] -= v;
                self.cell(i, j + 1, budget - v * pj, pj, c, f);
                self.col_rem[j] += v;
                self.diagonal[n] -= v;
            }
            v += 1;
            if v > max {
                break;
            }
        }
    }

    /// Complete the matrix with row 0, whose entries are what is left of the columns, and report
    /// its term.
    fn finish(&mut self, mut coef: u32, f: &mut impl FnMut(&[u32], u32)) {
        let mut t = [0u32; 2 * MAX_LEN];
        let len = self.r.len() + self.cols - 1;
        for n in 1..=len {
            let v = if n < self.cols { self.col_rem[n] } else { 0 };
            coef = self.place(n, v, coef);
            if coef == 0 {
                return;
            }
            t[n - 1] = self.diagonal[n] + v;
        }
        let mut len = len;
        while len > 0 && t[len - 1] == 0 {
            len -= 1;
        }
        f(&t[..len], coef);
    }
}

/// The number of interior entries (not in row or column 0) on anti-diagonals of index at most
/// [`MAX_LEN`].
const MAX_CELLS: usize = MAX_LEN * (MAX_LEN - 1) / 2;

/// Call `emit` with every $T$ such that $P(T)$ appears in $P(R) P(S)$ at the prime 2, with
/// multiplicity: equal terms may be emitted several times and should be summed mod 2.
///
/// This is the enumeration of [`algebra`]'s `antidiagonal` module on [`Seq`]s. Mod 2 a Milnor
/// matrix contributes exactly when the entries of each anti-diagonal have pairwise disjoint binary
/// digits. The interior entries are filled one anti-diagonal at a time, from the last one down.
/// The other entries of row $d$ and column $d$ lie on later anti-diagonals, so on reaching
/// anti-diagonal $d$ the slacks $x_{d0}$ and $x_{0d}$ are known. The anti-diagonal starts from
/// their bitwise or, and each interior entry only takes values disjoint from the bits used so far,
/// so every completed matrix contributes.
fn antidiagonal_terms(r: &Seq, s: &Seq, mut emit: impl FnMut(&[u32])) {
    let rows = r.len;
    let cols = s.len;
    // Entries on anti-diagonals d with 2^d - 1 above the polynomial degree of the product vanish.
    let degree: u64 = [r, s]
        .iter()
        .flat_map(|x| x.as_slice().iter().enumerate())
        .map(|(i, &e)| ((1u64 << (i + 1)) - 1) * e as u64)
        .sum();
    let mut max_diag = rows + cols;
    while max_diag > rows.max(cols) && (1u64 << max_diag) - 1 > degree {
        max_diag -= 1;
    }
    assert!(
        max_diag <= MAX_LEN,
        "product exceeds the supported length of Milnor sequences"
    );

    // What is left of `r` and `s`, indexed from 1. Once row (resp. column) `d` is filled, this is
    // its entry in column (resp. row) 0.
    let mut r_left = [0u32; MAX_LEN + 1];
    let mut s_left = [0u32; MAX_LEN + 1];
    r_left[1..=rows].copy_from_slice(r.as_slice());
    s_left[1..=cols].copy_from_slice(s.as_slice());

    // The interior cells in fill order, and whether each is the last of its anti-diagonal.
    let mut row = [0u8; MAX_CELLS];
    let mut col = [0u8; MAX_CELLS];
    let mut last = [false; MAX_CELLS];
    let mut num_cells = 0;
    for d in (2..=max_diag).rev() {
        for i in d.saturating_sub(cols).max(1)..=rows.min(d - 1) {
            row[num_cells] = i as u8;
            col[num_cells] = (d - i) as u8;
            num_cells += 1;
        }
        last[num_cells - 1] = true;
    }

    let mut t = [0u32; MAX_LEN];
    let (a, b) = (r_left[max_diag], s_left[max_diag]);
    if a & b != 0 {
        return;
    }
    if max_diag == 1 {
        return emit(&[a | b]);
    }

    // `value[k]` is the current entry of cell `k`, `bound[k]` the most it may be given the entries
    // before it, and `before[k]` the bits used on its anti-diagonal by the cells before it.
    let mut value = [0u32; MAX_CELLS];
    let mut bound = [0u32; MAX_CELLS];
    let mut before = [0u32; MAX_CELLS];

    let emit_t = |t: &mut [u32; MAX_LEN], first: u32, emit: &mut dyn FnMut(&[u32])| {
        t[0] = first;
        let mut len = max_diag;
        while len > 0 && t[len - 1] == 0 {
            len -= 1;
        }
        emit(&t[..len]);
    };

    let mut k = 0;
    let mut tally = a | b;
    'enter: loop {
        // Value 0 never overlaps.
        let (i, j) = (row[k] as usize, col[k] as usize);
        before[k] = tally;
        bound[k] = (r_left[i] >> j).min(s_left[j]);
        value[k] = 0;
        'advance: loop {
            let used = before[k] | value[k];
            if !last[k] {
                k += 1;
                tally = used;
                continue 'enter;
            }
            // Anti-diagonal `d` is complete, and the slacks of `d - 1` are now known.
            let d = (row[k] + col[k]) as usize;
            t[d - 1] = used;
            let (a, b) = (r_left[d - 1], s_left[d - 1]);
            if a & b == 0 {
                if d == 2 {
                    emit_t(&mut t, a | b, &mut emit);
                } else {
                    k += 1;
                    tally = a | b;
                    continue 'enter;
                }
            }
            // Backtrack to the latest cell that can still be increased.
            loop {
                let (i, j) = (row[k] as usize, col[k] as usize);
                let v = value[k];
                let mask = before[k];
                let next = ((v | mask) + 1) & !mask;
                if next <= bound[k] {
                    r_left[i] -= (next - v) << j;
                    s_left[j] -= next - v;
                    value[k] = next;
                    continue 'advance;
                }
                r_left[i] += v << j;
                s_left[j] += v;
                if k == 0 {
                    return;
                }
                k -= 1;
            }
        }
    }
}
