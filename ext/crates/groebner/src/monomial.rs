//! Packed Milnor basis elements and module monomials.

/// The bit offset of the May weight.
pub const WEIGHT_SHIFT: u32 = 112;
/// The bit offset of the exterior part.
pub const Q_SHIFT: u32 = 96;
/// The number of $Q_k$ a monomial can hold.
pub const Q_BITS: u32 = 16;
/// The number of bits holding the digits of the polynomial part.
pub const DIGIT_BITS: u32 = 96;

/// A Milnor basis element $Q(E) P(R)$, packed so that integer comparison is the monomial order.
///
/// From the most significant end: the May weight, the exterior part with $Q_0$ most significant,
/// and the p-adic digits of $R$ in the fields described by [`Params`](crate::Params). Operations
/// that depend on the field layout live on `Params`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mono(pub u128);

impl Mono {
    /// The unit.
    pub const ONE: Self = Self(0);
    /// The bits of the exterior part.
    pub const Q_MASK: u128 = ((1 << Q_BITS) - 1) << Q_SHIFT;
    /// The bits of the digits.
    pub const DIGIT_MASK: u128 = (1 << DIGIT_BITS) - 1;

    /// The exterior part in packed order, with $Q_k$ in bit `Q_BITS - 1 - k`.
    pub fn q_raw(self) -> u32 {
        ((self.0 & Self::Q_MASK) >> Q_SHIFT) as u32
    }

    /// Convert an exterior part with $Q_k$ in bit `k` to its packed form.
    pub fn q_part_to_raw(q_part: u32) -> u128 {
        ((q_part as u16).reverse_bits() as u128) << Q_SHIFT
    }

    /// The exterior part with $Q_k$ in bit `k`.
    pub fn q_part(self) -> u32 {
        (self.q_raw() as u16).reverse_bits() as u32
    }

    /// The indices $k$ of the $Q_k$ in this monomial, in increasing order.
    pub fn q_indices(self) -> impl Iterator<Item = u32> {
        let q = self.q_part();
        (0..Q_BITS).filter(move |k| q & (1 << k) != 0)
    }

    /// The number of $Q_k$, which is the exterior degree.
    pub fn ext_degree(self) -> u32 {
        self.q_raw().count_ones()
    }

    /// The May weight.
    pub fn weight(self) -> u32 {
        (self.0 >> WEIGHT_SHIFT) as u32
    }

    /// The digit fields.
    pub fn digits(self) -> u128 {
        self.0 & Self::DIGIT_MASK
    }

    /// Whether this is the unit.
    pub fn is_one(self) -> bool {
        self.0 == 0
    }
}

/// A monomial $m \cdot g$ of a free module, where `gen` indexes the generator $g$.
///
/// Larger generators come first, then monomials are compared as [`Mono`]s. The leading term of a
/// module element is its smallest term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModMono {
    /// The index of the generator.
    pub generator: u32,
    /// The algebra part.
    pub mono: Mono,
}

impl ModMono {
    /// The monomial `mono * generator`.
    pub fn new(generator: u32, mono: Mono) -> Self {
        Self { generator, mono }
    }
}

impl Ord for ModMono {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .generator
            .cmp(&self.generator)
            .then(self.mono.cmp(&other.mono))
    }
}

impl PartialOrd for ModMono {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
