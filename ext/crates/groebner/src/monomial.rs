//! Packed Milnor basis elements and module monomials.

/// The number of bits holding the digits of the polynomial part.
pub const DIGIT_BITS: u32 = 80;
/// The bit offset of the exterior part.
pub const Q_SHIFT: u32 = DIGIT_BITS;
/// The number of $Q_k$ a monomial can hold.
pub const Q_BITS: u32 = 14;
/// The bit offset of the May weight.
pub const WEIGHT_SHIFT: u32 = Q_SHIFT + Q_BITS;
/// The number of bits of the May weight, which is at most the degree.
pub const WEIGHT_BITS: u32 = 14;
/// The bit offset of the generator in a [`ModMono`].
pub const GENERATOR_SHIFT: u32 = WEIGHT_SHIFT + WEIGHT_BITS;
/// The number of bits of the generator in a [`ModMono`].
pub const GENERATOR_BITS: u32 = u128::BITS - GENERATOR_SHIFT;

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
        (((q_part as u16).reverse_bits() >> (u16::BITS - Q_BITS)) as u128) << Q_SHIFT
    }

    /// The exterior part with $Q_k$ in bit `k`.
    pub fn q_part(self) -> u32 {
        ((self.q_raw() as u16).reverse_bits() >> (u16::BITS - Q_BITS)) as u32
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
        ((self.0 >> WEIGHT_SHIFT) as u32) & ((1 << WEIGHT_BITS) - 1)
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

/// A monomial $m \cdot g$ of a free module, packed into one word with the complement of the index
/// of $g$ above the [`Mono`].
///
/// Integer comparison puts larger generators first, then compares the monomials. The leading term
/// of a module element is its smallest term.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModMono(u128);

impl ModMono {
    /// The bits of the generator.
    const GENERATOR_MASK: u32 = ((1u64 << GENERATOR_BITS) - 1) as u32;

    /// The monomial `mono * g`, where `generator` is the index of `g`.
    pub fn new(generator: u32, mono: Mono) -> Self {
        debug_assert!(generator <= Self::GENERATOR_MASK, "too many generators");
        debug_assert!(mono.0 >> GENERATOR_SHIFT == 0);
        Self((((!generator & Self::GENERATOR_MASK) as u128) << GENERATOR_SHIFT) | mono.0)
    }

    /// The index of the generator.
    pub fn generator(self) -> u32 {
        !((self.0 >> GENERATOR_SHIFT) as u32) & Self::GENERATOR_MASK
    }

    /// The algebra part.
    pub fn mono(self) -> Mono {
        Mono(self.0 & ((1 << GENERATOR_SHIFT) - 1))
    }
}

impl std::fmt::Debug for ModMono {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} g{}", self.mono(), self.generator())
    }
}
