//! Critical pairs, bucketed by degree and pruned by the Gebauer–Möller criteria.

use std::collections::BTreeMap;

use rustc_hash::FxHashSet;

use crate::{ModMono, Mono, Params};

/// The index `i1` of a critical single.
pub const SINGLE: u32 = u32::MAX;

/// A critical pair $m_1 g_{i_1} - m_2 g_{i_2}$, up to scalars, or a critical single $m_2 g_{i_2}$
/// with $m_2$ annihilating the leading term of $g_{i_2}$ in the associated graded algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The first element, or [`SINGLE`].
    pub i1: u32,
    /// The second element, which is the later of the two.
    pub i2: u32,
    /// The multiplier of the first element.
    pub m1: Mono,
    /// The multiplier of the second element.
    pub m2: Mono,
    removed: bool,
}

impl Pair {
    /// Whether this is a critical single.
    pub fn is_single(&self) -> bool {
        self.i1 == SINGLE
    }
}

/// How many pairs were created and pruned in one degree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairCounts {
    /// Pairs formed between leading terms on the same generator.
    pub formed: u64,
    /// Pairs dropped because their multiplier is divisible by that of another new pair.
    pub divisible: u64,
    /// Pairs removed by the chain criterion.
    pub chain: u64,
    /// Critical singles.
    pub singles: u64,
}

/// The critical pairs of a Gröbner basis that is built up one leading term at a time.
///
/// `syzygies[j]` holds the pairs $(i, j)$ whose multipliers $m_2$ minimally generate the monomial
/// ideal $(L_0, \dots, L_{j-1}) : L_j$ of the associated graded algebra. Pairs not yet reduced are
/// kept in buckets by degree, and pairs that the chain criterion shows to be redundant are removed
/// from their bucket just before it is used.
#[derive(Debug)]
pub struct PairStore {
    t_trunc: i32,
    syzygies: Vec<Vec<Pair>>,
    /// `min_pairs[t]` are the pairs of degree `t`, sorted by `i2` because leads are added in
    /// order.
    min_pairs: BTreeMap<i32, Vec<Pair>>,
    redundant: BTreeMap<i32, FxHashSet<(u32, u32)>>,
    singles: BTreeMap<i32, Vec<Pair>>,
    counts: BTreeMap<i32, PairCounts>,
}

impl PairStore {
    /// An empty store that ignores pairs of degree above `t_trunc`.
    pub fn new(t_trunc: i32) -> Self {
        Self {
            t_trunc,
            syzygies: Vec::new(),
            min_pairs: BTreeMap::new(),
            redundant: BTreeMap::new(),
            singles: BTreeMap::new(),
            counts: BTreeMap::new(),
        }
    }

    /// Rebuild the store for the leading terms `leads` with a new truncation degree, keeping only
    /// pairs of degree at least `t_min`.
    pub fn rebuild(
        params: &Params,
        leads: &[ModMono],
        generator_degree: impl Fn(u32) -> i32,
        t_trunc: i32,
        t_min: i32,
    ) -> Self {
        let mut store = Self::new(t_trunc);
        for (n, &lead) in leads.iter().enumerate() {
            store.add_lead(
                params,
                &leads[..n],
                lead,
                generator_degree(lead.generator()),
            );
        }
        store.min_pairs = store.min_pairs.split_off(&t_min);
        store.redundant = store.redundant.split_off(&t_min);
        store.singles = store.singles.split_off(&t_min);
        store.counts = store.counts.split_off(&t_min);
        store
    }

    /// The truncation degree.
    pub fn t_trunc(&self) -> i32 {
        self.t_trunc
    }

    /// Register the new leading term `lead`, of a generator of degree `generator_degree`, after
    /// the existing `leads`.
    pub fn add_lead(
        &mut self,
        params: &Params,
        leads: &[ModMono],
        lead: ModMono,
        generator_degree: i32,
    ) {
        let n = leads.len();
        if self.syzygies.len() < n + 1 {
            self.syzygies.resize(n + 1, Vec::new());
        }

        let mut new_pairs: Vec<Option<(i32, Pair)>> = leads
            .iter()
            .enumerate()
            .map(|(i, l)| {
                if l.generator() != lead.generator() {
                    return None;
                }
                let lcm = params.lcm(l.mono(), lead.mono());
                let t = params.degree(lcm) + generator_degree;
                (t <= self.t_trunc).then(|| {
                    let pair = Pair {
                        i1: i as u32,
                        i2: n as u32,
                        m1: params.quotient(lcm, l.mono()),
                        m2: params.quotient(lcm, lead.mono()),
                        removed: false,
                    };
                    (t, pair)
                })
            })
            .collect();
        for (t, _) in new_pairs.iter().flatten() {
            self.counts.entry(*t).or_default().formed += 1;
        }
        let formed_degrees: Vec<Option<i32>> =
            new_pairs.iter().map(|p| p.map(|(t, _)| t)).collect();
        let mut chain_marked: Vec<(i32, u32, u32)> = Vec::new();

        for j in 1..n {
            let Some((_, pj)) = new_pairs[j] else {
                continue;
            };
            for i in 0..j {
                let Some((_, pi)) = new_pairs[i] else {
                    continue;
                };
                if params.divides(pi.m2, pj.m2) {
                    new_pairs[j] = None;
                    break;
                } else if params.divides(pj.m2, pi.m2) {
                    new_pairs[i] = None;
                } else if params.coprime(pi.m1, pj.m1) {
                    // L_n divides lcm(L_i, L_j), so (i, j) is generated by (i, n) and (j, n).
                    let lcm = params.lcm(leads[i].mono(), leads[j].mono());
                    let t = params.degree(lcm) + generator_degree;
                    if t <= self.t_trunc {
                        chain_marked.push((t, i as u32, j as u32));
                    }
                }
            }
        }
        for (t, i, j) in chain_marked {
            self.redundant.entry(t).or_default().insert((i, j));
        }

        for (pair, t) in new_pairs.into_iter().zip(formed_degrees) {
            let Some((t, pair)) = pair else {
                if let Some(t) = t {
                    self.counts.entry(t).or_default().divisible += 1;
                }
                continue;
            };
            self.syzygies[n].push(pair);
            self.min_pairs.entry(t).or_default().push(pair);
        }

        let t_lead = params.degree(lead.mono()) + generator_degree;
        for m in params.annihilators(lead.mono()) {
            let t = t_lead + params.degree(m);
            if t <= self.t_trunc {
                self.singles.entry(t).or_default().push(Pair {
                    i1: SINGLE,
                    i2: n as u32,
                    m1: Mono::ONE,
                    m2: m,
                    removed: false,
                });
                self.counts.entry(t).or_default().singles += 1;
            }
        }
    }

    /// Remove the pairs of degree `t` that the chain criterion shows to be redundant.
    fn minimize(&mut self, params: &Params, leads: &[ModMono], t: i32) {
        let Some(redundant) = self.redundant.remove(&t) else {
            return;
        };
        let Some(bucket) = self.min_pairs.get_mut(&t) else {
            return;
        };
        let mut removed = 0;
        for (i, j) in redundant {
            let (mut i, mut j) = (i as usize, j as usize);
            loop {
                let gcd = params.gcd(leads[i].mono(), leads[j].mono());
                let m2 = params.quotient(leads[i].mono(), gcd);
                let start = bucket.partition_point(|c| (c.i2 as usize) < j);
                if let Some(pair) = bucket[start..]
                    .iter_mut()
                    .take_while(|c| c.i2 as usize == j)
                    .find(|c| c.m2 == m2 && !c.removed)
                {
                    pair.removed = true;
                    removed += 1;
                    break;
                }
                // Rewrite (i, j) in terms of the minimal syzygies (k, j).
                let Some(c) = self.syzygies[j].iter().find(|c| params.divides(c.m2, m2)) else {
                    debug_assert!(false, "the syzygies of {j} do not generate ({i}, {j})");
                    break;
                };
                let m1 = params.quotient(leads[j].mono(), gcd);
                if !params.coprime(c.m1, m1) {
                    break;
                }
                let k = c.i1 as usize;
                (i, j) = (i.min(k), i.max(k));
            }
        }
        self.counts.entry(t).or_default().chain += removed;
    }

    /// Take the pairs and singles of degree `t`, after pruning them with the chain criterion.
    pub fn take(&mut self, params: &Params, leads: &[ModMono], t: i32) -> (Vec<Pair>, PairCounts) {
        self.minimize(params, leads, t);
        let mut result = self.singles.remove(&t).unwrap_or_default();
        if let Some(bucket) = self.min_pairs.remove(&t) {
            result.extend(bucket.into_iter().filter(|c| !c.removed));
        }
        // Buckets below `t` can only be left over from degrees that were skipped.
        self.min_pairs = self.min_pairs.split_off(&t);
        self.redundant = self.redundant.split_off(&t);
        self.singles = self.singles.split_off(&t);
        let counts = self.counts.remove(&t).unwrap_or_default();
        self.counts = self.counts.split_off(&t);
        (result, counts)
    }
}
