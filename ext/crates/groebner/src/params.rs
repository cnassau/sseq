//! The Steenrod algebra a Gröbner resolution works over, and the layout of its packed monomials.

use algebra::SteenrodKind;

use crate::monomial::{DIGIT_BITS, Mono, Q_BITS, WEIGHT_BITS, WEIGHT_SHIFT};

/// Degrees of positions are capped here so that sums of degrees stay well inside an `i32`.
const DEGREE_CAP: i64 = 1 << 24;

/// A digit position of a packed monomial: the exponent of $\xi_n^{p^i}$ in the dual basis.
///
/// A Milnor basis element $P(R)$ has $r_n = \sum_i d_{i,n} p^i$, and the digit $d_{i,n}$ is stored
/// in the position with `exp = i` and `xi = n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    /// The exponent $i$ in $\xi_n^{p^i}$.
    pub exp: u32,
    /// The index $n$ in $\xi_n^{p^i}$.
    pub xi: u32,
    /// The degree of $\xi_n^{p^i}$.
    pub degree: i32,
    /// The May weight $2n - 1$ of one unit of this digit.
    pub weight: u32,
    /// The bit offset of the least significant bit of the digit field.
    pub shift: u32,
    /// $p^i$, the contribution of one unit of this digit to $r_n$.
    pub value: u32,
}

/// The prime, kind and monomial encoding of a Steenrod algebra.
///
/// Monomials are ordered by May weight first, then by their exterior part with $Q_0$ most
/// significant, then by their digits read as one integer with the lowest-degree position most
/// significant, as [`Mono`] lays them out. The leading term of a polynomial is its *smallest* term.
/// This is Lin's order at `p = 2`, and makes the leading term of a product the product of the
/// leading terms in the associated graded algebra
/// $E[Q_0, Q_1, \dots] \otimes \bigotimes_{i,n} \mathbb{F}_p[x_{i,n}]/(x_{i,n}^p)$ whenever that
/// is non-zero.
#[derive(Debug)]
pub struct Params {
    p: u32,
    kind: SteenrodKind,
    has_exterior: bool,
    q: i32,
    /// Bits per digit field. At odd primes the top bit of each field is a guard bit that is zero in
    /// every valid monomial, which lets digitwise comparisons run on the whole word at once.
    width: u32,
    positions: Vec<Position>,
    /// `position_of[n - 1][i]` is the index of the position of $\xi_n^{p^i}$.
    position_of: Vec<Vec<usize>>,
    tau_degrees: Vec<i32>,
    max_degree: i32,
    guard: u128,
    /// $2^{w-1} - p$ in each field, where $w$ is the field width.
    carry_offset: u128,
    /// $2^{w-1} - 1$ in each field.
    nonzero_offset: u128,
    inverses: Vec<u32>,
    factorials: Vec<u32>,
    inverse_factorials: Vec<u32>,
    /// `encodings[n - 1][r]` is $P(0, \dots, 0, r)$ with $r$ in position $n$, for every $r$ whose
    /// degree is at most `max_degree`. Encodings of different $n$ add up to that of $P(R)$.
    encodings: Vec<Vec<u128>>,
    /// At `p = 2`, the position index and value of the digit at each bit.
    bit_positions: Vec<(u8, u32)>,
}

impl Params {
    /// The parameters of the Steenrod algebra of the given kind at the prime `p`.
    ///
    /// # Panics
    ///
    /// If `p` is not a prime or the kind does not exist at `p`.
    pub fn new(p: u32, kind: SteenrodKind) -> Self {
        assert!(
            p >= 2 && (2..p).all(|d| !p.is_multiple_of(d)),
            "{p} is not a prime"
        );
        assert!(
            p <= crate::poly::MAX_PRIME,
            "coefficients mod {p} do not fit a Poly"
        );
        let (has_exterior, q) = match kind {
            SteenrodKind::Classical if p == 2 => (false, 1),
            SteenrodKind::Classical => (true, 2 * (p as i32 - 1)),
            SteenrodKind::MotivicModTau => {
                assert_eq!(p, 2, "A_C/tau only exists at p = 2");
                (true, 2)
            }
            SteenrodKind::ReducedPowers => {
                assert!(p != 2, "the reduced powers only exist at odd primes");
                (false, 2 * (p as i32 - 1))
            }
        };
        let width = if p == 2 {
            1
        } else {
            u32::BITS - (p - 1).leading_zeros() + 1
        };
        let capacity = (DIGIT_BITS / width) as usize;

        let p64 = p as i64;
        let xi_degree = |n: u32| q as i64 * (p64.pow(n) - 1) / (p64 - 1);
        let mut positions = Vec::new();
        let mut position_of: Vec<Vec<usize>> = Vec::new();
        let mut max_degree = DEGREE_CAP as i32;
        'outer: for j in 1u32.. {
            for exp in (0..j).rev() {
                let xi = j - exp;
                let degree = p64
                    .checked_pow(exp)
                    .and_then(|pe| pe.checked_mul(xi_degree(xi)))
                    .filter(|&d| d <= DEGREE_CAP);
                let Some(degree) = degree.filter(|_| positions.len() < capacity) else {
                    // Positions come in increasing degree, so everything below this one fits.
                    max_degree = degree.map_or(DEGREE_CAP, |d| d - 1) as i32;
                    break 'outer;
                };
                let idx = positions.len();
                positions.push(Position {
                    exp,
                    xi,
                    degree: degree as i32,
                    weight: 2 * xi - 1,
                    shift: DIGIT_BITS - (idx as u32 + 1) * width,
                    value: p.pow(exp),
                });
                if position_of.len() < xi as usize {
                    position_of.resize(xi as usize, Vec::new());
                }
                let row = &mut position_of[xi as usize - 1];
                assert_eq!(row.len(), exp as usize);
                row.push(idx);
            }
        }

        // The weight of a monomial is at most its degree.
        max_degree = max_degree.min((1 << WEIGHT_BITS) - 1);
        assert!(
            position_of.len() <= crate::product::MAX_LEN,
            "positions exceed the supported length of Milnor sequences"
        );

        let tau_degrees: Vec<i32> = if has_exterior {
            (0..Q_BITS)
                .map(|k| p64.checked_pow(k).map_or(i64::MAX, |pk| 2 * pk - 1))
                .map(|d| d.min(DEGREE_CAP + 1) as i32)
                .collect()
        } else {
            Vec::new()
        };
        if has_exterior {
            // The degree of Q_{Q_BITS}, which is not representable.
            let next = p64.saturating_pow(Q_BITS).saturating_mul(2) - 1;
            max_degree = max_degree.min(next.min(DEGREE_CAP) as i32);
        }

        let mut guard = 0;
        let mut carry_offset = 0;
        let mut nonzero_offset = 0;
        for pos in &positions {
            if p != 2 {
                guard |= 1u128 << (pos.shift + width - 1);
                carry_offset |= ((1u128 << (width - 1)) - p as u128) << pos.shift;
                nonzero_offset |= ((1u128 << (width - 1)) - 1) << pos.shift;
            }
        }

        let mut inverses = vec![0; p as usize];
        let mut factorials = vec![1; p as usize];
        let mut inverse_factorials = vec![1; p as usize];
        for a in 1..p {
            inverses[a as usize] = (1..p).find(|b| a * b % p == 1).unwrap();
            factorials[a as usize] = factorials[a as usize - 1] * a % p;
        }
        for a in 0..p {
            inverse_factorials[a as usize] = inverses[factorials[a as usize] as usize];
        }

        let mut params = Self {
            p,
            kind,
            has_exterior,
            q,
            width,
            positions,
            position_of,
            tau_degrees,
            max_degree,
            guard,
            carry_offset,
            nonzero_offset,
            inverses,
            factorials,
            inverse_factorials,
            encodings: Vec::new(),
            bit_positions: Vec::new(),
        };
        params.encodings = (1..=params.position_of.len() as u32)
            .map(|n| {
                let degree = params.positions[params.position_of[n as usize - 1][0]].degree;
                (0..=(params.max_degree / degree) as u32)
                    .map(|r| params.encode_slow(n, r).0)
                    .collect()
            })
            .collect();
        if p == 2 {
            params.bit_positions = vec![(0, 0); DIGIT_BITS as usize];
            for pos in &params.positions {
                params.bit_positions[pos.shift as usize] = ((pos.xi - 1) as u8, pos.value);
            }
        }
        params
    }

    /// The monomial $P(0, \dots, 0, r)$ with $r$ in position $n$, digit by digit.
    fn encode_slow(&self, n: u32, r: u32) -> Mono {
        let mut digits = Vec::new();
        let mut rest = r;
        let mut exp = 0;
        while rest > 0 {
            let d = rest % self.p;
            if d != 0 {
                digits.push((self.position_of(n, exp).unwrap(), d));
            }
            rest /= self.p;
            exp += 1;
        }
        self.from_digits(0, digits)
    }

    /// The prime.
    pub fn prime(&self) -> u32 {
        self.p
    }

    /// The kind of Steenrod algebra.
    pub fn kind(&self) -> SteenrodKind {
        self.kind
    }

    /// Whether the algebra contains the exterior generators $Q_k$.
    pub fn has_exterior(&self) -> bool {
        self.has_exterior
    }

    /// The degree of $P^1$ (or $Sq^1$ in the classical case at `p = 2`).
    pub fn q(&self) -> i32 {
        self.q
    }

    /// Every monomial of at most this degree is representable.
    pub fn max_degree(&self) -> i32 {
        self.max_degree
    }

    /// The digit positions, in order of decreasing significance.
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// The index of the position of $\xi_n^{p^i}$, if it is representable.
    pub fn position_of(&self, xi: u32, exp: u32) -> Option<usize> {
        self.position_of
            .get(xi.checked_sub(1)? as usize)?
            .get(exp as usize)
            .copied()
    }

    /// The degree of $Q_k$.
    pub fn tau_degree(&self, k: u32) -> i32 {
        self.tau_degrees[k as usize]
    }

    /// The May weight of $Q_k$.
    pub fn tau_weight(k: u32) -> u32 {
        2 * k + 1
    }

    /// The inverse of a non-zero residue mod p.
    pub fn inverse(&self, c: u32) -> u32 {
        debug_assert!(!c.is_multiple_of(self.p));
        self.inverses[(c % self.p) as usize]
    }

    /// The negative of a residue mod p.
    pub fn neg(&self, c: u32) -> u32 {
        (self.p - c % self.p) % self.p
    }

    /// The multinomial coefficient $\binom{\sum x}{x_1, x_2, \dots}$ mod p, computed digitwise by
    /// Lucas' theorem.
    pub fn multinomial(&self, xs: &mut [u32]) -> u32 {
        let p = self.p;
        let mut coef = 1;
        loop {
            let mut sum = 0;
            let mut any = false;
            for x in xs.iter_mut() {
                let d = *x % p;
                *x /= p;
                any |= *x != 0;
                sum += d;
                if sum >= p {
                    return 0;
                }
                coef = coef * self.inverse_factorials[d as usize] % p;
            }
            coef = coef * self.factorials[sum as usize] % p;
            if !any {
                return coef;
            }
        }
    }

    /// The digit of `m` in position `idx`.
    pub fn digit(&self, m: Mono, idx: usize) -> u32 {
        ((m.0 >> self.positions[idx].shift) & ((1 << self.width) - 1)) as u32
    }

    /// Iterate over the non-zero digits of `m` as `(position index, digit)`.
    pub fn digits(&self, m: Mono) -> impl Iterator<Item = (usize, u32)> + '_ {
        let mut rest = m.digits();
        std::iter::from_fn(move || {
            if rest == 0 {
                return None;
            }
            let top = DIGIT_BITS - 1 - (127 - rest.leading_zeros());
            let idx = (top / self.width) as usize;
            let d = self.digit(Mono(rest), idx);
            rest &= !(((1u128 << self.width) - 1) << self.positions[idx].shift);
            Some((idx, d))
        })
    }

    /// The monomial with exterior part `q_part` (bit `k` standing for $Q_k$) and the digit `d` in
    /// each `(idx, d)` of `digits`.
    pub fn from_digits(&self, q_part: u32, digits: impl IntoIterator<Item = (usize, u32)>) -> Mono {
        debug_assert!(q_part < 1 << Q_BITS);
        let mut raw = Mono::q_part_to_raw(q_part);
        let mut weight: u128 = (0..Q_BITS)
            .filter(|k| q_part & (1 << k) != 0)
            .map(|k| Self::tau_weight(k) as u128)
            .sum();
        for (idx, d) in digits {
            debug_assert!(d < self.p);
            let pos = &self.positions[idx];
            raw |= (d as u128) << pos.shift;
            weight += (d * pos.weight) as u128;
        }
        Mono(raw | (weight << WEIGHT_SHIFT))
    }

    /// The monomial $Q(E) P(R)$, where bit `k` of `q_part` stands for $Q_k$. Returns `None` if it
    /// is not representable.
    pub fn encode(&self, q_part: u32, r: &[u32]) -> Option<Mono> {
        if q_part >= 1 << Q_BITS {
            return None;
        }
        let mut raw = 0;
        if q_part != 0 {
            let weight: u32 = (0..Q_BITS)
                .filter(|k| q_part & (1 << k) != 0)
                .map(Self::tau_weight)
                .sum();
            raw = Mono::q_part_to_raw(q_part) | ((weight as u128) << WEIGHT_SHIFT);
        }
        for (n, &rn) in r.iter().enumerate() {
            if rn != 0 {
                raw += *self.encodings.get(n)?.get(rn as usize)?;
            }
        }
        Some(Mono(raw))
    }

    /// The position index and value $p^i$ of each non-zero digit unit of `m` at `p = 2`, or
    /// `None` at odd primes.
    pub(crate) fn bit_positions(&self) -> Option<&[(u8, u32)]> {
        (self.p == 2).then_some(&self.bit_positions[..])
    }

    /// The exterior part and the sequence $R$ of $Q(E) P(R)$, without trailing zeros.
    pub fn decode(&self, m: Mono) -> (u32, Vec<u32>) {
        let mut r = Vec::new();
        for (idx, d) in self.digits(m) {
            let pos = &self.positions[idx];
            let n = pos.xi as usize;
            if r.len() < n {
                r.resize(n, 0);
            }
            r[n - 1] += d * pos.value;
        }
        (m.q_part(), r)
    }

    /// The internal degree of `m`.
    pub fn degree(&self, m: Mono) -> i32 {
        let q_deg: i32 = m.q_indices().map(|k| self.tau_degree(k)).sum();
        q_deg
            + self
                .digits(m)
                .map(|(idx, d)| d as i32 * self.positions[idx].degree)
                .sum::<i32>()
    }

    /// The May filtration of `m`: each unit of a digit of $\xi_n^{p^i}$ counts $n$, and $Q_k$
    /// counts $k + 1$.
    ///
    /// The May weight of the monomial order counts $2n - 1$ and $2k + 1$ instead, so this is half
    /// of the weight plus the number of digit units and exterior generators.
    pub fn may_filtration(&self, m: Mono) -> u32 {
        let units = if self.p == 2 {
            m.digits().count_ones()
        } else {
            self.digits(m).map(|(_, d)| d).sum()
        };
        (m.weight() + units + m.ext_degree()) / 2
    }

    /// The monomial with the same digits and exterior part as `m` and the correct weight.
    fn reweigh(&self, m: Mono) -> Mono {
        let raw = Mono(m.0 & !(u128::MAX << WEIGHT_SHIFT));
        self.from_digits(raw.q_part(), self.digits(raw))
    }

    /// The value bits of the fields of `a` that are at least the corresponding fields of `b`.
    fn ge_mask(&self, a: u128, b: u128) -> u128 {
        let g = ((a | self.guard) - b) & self.guard;
        g - (g >> (self.width - 1))
    }

    /// Whether `a` divides `b` in the associated graded algebra.
    pub fn divides(&self, a: Mono, b: Mono) -> bool {
        let (qa, qb) = (a.q_raw(), b.q_raw());
        if qa & !qb != 0 {
            return false;
        }
        let (da, db) = (a.digits(), b.digits());
        if self.p == 2 {
            da & !db == 0
        } else {
            ((db | self.guard) - da) & self.guard == self.guard
        }
    }

    /// The quotient `a / b` in the associated graded algebra. Requires `b` to divide `a`.
    pub fn quotient(&self, a: Mono, b: Mono) -> Mono {
        debug_assert!(self.divides(b, a), "{a:?} is not divisible by {b:?}");
        Mono(a.0 - b.0)
    }

    /// The product `a * b` in the associated graded algebra, up to a non-zero scalar, or `None` if
    /// it vanishes.
    pub fn lf_mul(&self, a: Mono, b: Mono) -> Option<Mono> {
        if a.q_raw() & b.q_raw() != 0 {
            return None;
        }
        let (da, db) = (a.digits(), b.digits());
        let carry = if self.p == 2 {
            da & db != 0
        } else {
            (da + db + self.carry_offset) & self.guard != 0
        };
        (!carry).then_some(Mono(a.0 + b.0))
    }

    /// The digitwise minimum, i.e. the greatest common divisor in the associated graded algebra.
    pub fn gcd(&self, a: Mono, b: Mono) -> Mono {
        let (da, db) = (a.digits(), b.digits());
        let digits = if self.p == 2 {
            da & db
        } else {
            let m = self.ge_mask(da, db);
            (db & m) | (da & !m)
        };
        let q = (a.0 & b.0) & Mono::Q_MASK;
        self.reweigh(Mono(q | digits))
    }

    /// The digitwise maximum, i.e. the least common multiple in the associated graded algebra.
    pub fn lcm(&self, a: Mono, b: Mono) -> Mono {
        let (da, db) = (a.digits(), b.digits());
        let digits = if self.p == 2 {
            da | db
        } else {
            let m = self.ge_mask(da, db);
            (da & m) | (db & !m)
        };
        let q = (a.0 | b.0) & Mono::Q_MASK;
        self.reweigh(Mono(q | digits))
    }

    /// Whether `a` and `b` have no common divisor other than 1.
    pub fn coprime(&self, a: Mono, b: Mono) -> bool {
        if a.q_raw() & b.q_raw() != 0 {
            return false;
        }
        let (da, db) = (a.digits(), b.digits());
        if self.p == 2 {
            da & db == 0
        } else {
            let nz = |x: u128| (x + self.nonzero_offset) & self.guard;
            nz(da) & nz(db) == 0
        }
    }

    /// The monomials $x$ generating the annihilator of `m` in the associated graded algebra: one
    /// $x_{i,n}^{p - d}$ for every digit $d$ of `m` and one $Q_k$ for every $Q_k$ in `m`.
    pub fn annihilators(&self, m: Mono) -> impl Iterator<Item = Mono> + '_ {
        let digits = self
            .digits(m)
            .map(move |(idx, d)| self.from_digits(0, [(idx, self.p - d)]));
        let qs = m.q_indices().map(move |k| self.from_digits(1 << k, []));
        digits.chain(qs)
    }

    /// The coefficient $c$ with $a b = c \cdot \mathrm{lf\_mul}(a, b)$ in the associated graded
    /// algebra. Zero if the product vanishes there.
    pub fn lf_coefficient(&self, a: Mono, b: Mono) -> u32 {
        if self.lf_mul(a, b).is_none() {
            return 0;
        }
        let mut coef = 1;
        for (idx, d) in self.digits(a) {
            let e = self.digit(b, idx);
            coef = coef * self.multinomial(&mut [d, e]) % self.p;
        }
        // Q(E) Q(F) = (-1)^{#{(e, f) in E x F : e > f}} Q(E ∪ F).
        let (ea, eb) = (a.q_part(), b.q_part());
        let swaps: u32 = (0..Q_BITS)
            .filter(|&f| eb & (1 << f) != 0)
            .map(|f| (ea >> (f + 1)).count_ones())
            .sum();
        if swaps % 2 == 1 { self.neg(coef) } else { coef }
    }

    /// Format `m` as $Q(E) P(R)$.
    pub fn format(&self, m: Mono) -> String {
        let (q, r) = self.decode(m);
        let mut s = String::new();
        if q != 0 {
            let e: Vec<String> = (0..Q_BITS)
                .filter(|k| q & (1 << k) != 0)
                .map(|k| k.to_string())
                .collect();
            s += &format!("Q({})", e.join(","));
        }
        if !r.is_empty() {
            let r: Vec<String> = r.iter().map(|x| x.to_string()).collect();
            s += &format!("P({})", r.join(","));
        }
        if s.is_empty() {
            s.push('1');
        }
        s
    }
}
