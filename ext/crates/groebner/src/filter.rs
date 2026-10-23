//! Criteria that discard critical pairs before they are reduced.

use std::{collections::BTreeMap, fmt};

use crate::{ModMono, Mono, Params, pairs::Pair, resolution::Generator};

/// Everything a [`PairFilter`] may base its decision on.
#[derive(Clone, Copy, Debug)]
pub struct PairContext<'a> {
    /// The algebra.
    pub params: &'a Params,
    /// The level: the pair lives in $C_s \oplus C_{s+1}$.
    pub s: u32,
    /// The internal degree of the pair.
    pub t: i32,
    /// The pair.
    pub pair: &'a Pair,
    /// The leading term of the first element, absent for singles.
    pub lead1: Option<ModMono>,
    /// The leading term of the second element.
    pub lead2: ModMono,
    /// The generator of $C_s$ both leading terms lie on.
    pub generator: Generator,
}

impl PairContext<'_> {
    /// The leading monomial the pair cancels in the associated graded algebra: the least common
    /// multiple of the leading terms for a pair, and `None` for a single, whose product vanishes
    /// there.
    pub fn lcm(&self) -> Option<Mono> {
        self.lead1
            .map(|l1| self.params.lcm(l1.mono(), self.lead2.mono()))
    }

    /// The degree of the pair above its generator, which is the degree of the algebra elements
    /// involved.
    pub fn relative_degree(&self) -> i32 {
        self.t - self.generator.degree
    }

    /// The smallest $j$ such that every monomial of the pair lies in the span of the $\xi_n^{p^i}$
    /// with $i + n \leq j$ and the $\tau_k$ with $k < j$.
    ///
    /// The algebras with these duals are the $A(j - 1)$ at the prime 2, so this measures how deep
    /// into the Steenrod algebra the pair reaches.
    pub fn depth(&self) -> u32 {
        let monos = [
            self.lead1.map(|l| l.mono()),
            Some(self.lead2.mono()),
            Some(self.pair.m1),
            Some(self.pair.m2),
        ];
        monos
            .into_iter()
            .flatten()
            .map(|m| mono_depth(self.params, m))
            .max()
            .unwrap_or(0)
    }
}

/// The smallest $j$ with every $\xi_n^{p^i}$ of `m` satisfying $i + n \leq j$ and every $\tau_k$
/// satisfying $k < j$.
pub fn mono_depth(params: &Params, m: Mono) -> u32 {
    let p_depth = params
        .digits(m)
        .map(|(idx, _)| {
            let pos = &params.positions()[idx];
            pos.exp + pos.xi
        })
        .max()
        .unwrap_or(0);
    let q_depth = m.q_indices().map(|k| k + 1).max().unwrap_or(0);
    p_depth.max(q_depth)
}

/// A criterion predicting that a critical pair is redundant, in the sense of
/// [`Outcome::is_redundant`](crate::Outcome::is_redundant).
pub trait PairFilter: Send + Sync {
    /// A short name, used in statistics and logs.
    fn name(&self) -> &str;

    /// Whether the pair is predicted to be redundant.
    fn skip(&self, ctx: &PairContext) -> bool;
}

/// What happens to the pairs a filter predicts to be redundant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// Discard them. This changes the result if the prediction is wrong.
    Enforce,
    /// Reduce them anyway and record whether the prediction was right.
    Audit,
}

/// How a filter's predictions compare to the outcomes, on one level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterCounts {
    /// Pairs discarded in [`FilterMode::Enforce`].
    pub discarded: u64,
    /// Pairs predicted redundant that were.
    pub correct: u64,
    /// Pairs predicted redundant that were not. In [`FilterMode::Enforce`] these cannot be seen.
    pub wrong: u64,
    /// Redundant pairs that were not predicted.
    pub missed: u64,
}

/// A filter together with its mode and record.
pub(crate) struct InstalledFilter {
    pub(crate) filter: Box<dyn PairFilter>,
    pub(crate) mode: FilterMode,
    pub(crate) counts: BTreeMap<u32, FilterCounts>,
}

/// The record of every installed filter, by level.
#[derive(Clone, Debug, Default)]
pub struct FilterReport {
    /// `(name, mode, counts by level)` for each filter, in installation order.
    pub filters: Vec<(String, FilterMode, BTreeMap<u32, FilterCounts>)>,
}

impl fmt::Display for FilterReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, mode, by_level) in &self.filters {
            writeln!(f, "filter {name} ({mode:?})")?;
            writeln!(
                f,
                "{:>5} {:>10} {:>10} {:>10} {:>10}",
                "s", "discarded", "correct", "wrong", "missed"
            )?;
            let mut total = FilterCounts::default();
            for (s, c) in by_level {
                writeln!(
                    f,
                    "{s:>5} {:>10} {:>10} {:>10} {:>10}",
                    c.discarded, c.correct, c.wrong, c.missed
                )?;
                total.discarded += c.discarded;
                total.correct += c.correct;
                total.wrong += c.wrong;
                total.missed += c.missed;
            }
            writeln!(
                f,
                "{:>5} {:>10} {:>10} {:>10} {:>10}",
                "all", total.discarded, total.correct, total.wrong, total.missed
            )?;
        }
        Ok(())
    }
}

/// A filter given by a closure, for quick experiments.
pub struct FnFilter<F> {
    name: String,
    f: F,
}

impl<F: Fn(&PairContext) -> bool + Send + Sync> FnFilter<F> {
    /// The filter named `name` that skips a pair when `f` says so.
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

impl<F: Fn(&PairContext) -> bool + Send + Sync> PairFilter for FnFilter<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn skip(&self, ctx: &PairContext) -> bool {
        (self.f)(ctx)
    }
}
