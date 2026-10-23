//! Counters and per-pair event logs for experimenting with the algorithm.

use std::{collections::BTreeMap, fmt, io::Write};

use crate::{ModMono, filter::PairContext, pairs::PairCounts};

/// What reducing a critical pair, single or module relation produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Both components reduced to zero: the pair was redundant.
    Useless,
    /// The $C_s$ component survived and became a new element of the Gröbner basis.
    NewElement,
    /// The $C_s$ component vanished and the cycle in $C_{s+1}$ was a boundary of another cycle
    /// found in the same degree.
    Boundary,
    /// The $C_s$ component vanished and the cycle in $C_{s+1}$ became a new generator of
    /// $C_{s+2}$.
    NewGenerator,
    /// The $C_s$ component vanished, and the cycle would give a generator beyond the largest
    /// homological degree. Whether it is a boundary is not known.
    Beyond,
}

impl Outcome {
    /// Whether the reduction added nothing, so that skipping it would not have changed the
    /// resolution.
    pub fn is_redundant(self) -> bool {
        matches!(self, Self::Useless | Self::Boundary | Self::Beyond)
    }

    /// A short name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Useless => "useless",
            Self::NewElement => "element",
            Self::Boundary => "boundary",
            Self::NewGenerator => "generator",
            Self::Beyond => "beyond",
        }
    }
}

/// Everything known about one reduction.
#[derive(Clone, Copy, Debug)]
pub struct PairEvent<'a> {
    /// The level, i.e. the pair lives in $C_s \oplus C_{s+1}$. This is `-1` for the relations of
    /// the module.
    pub s: i32,
    /// The internal degree.
    pub t: i32,
    /// The pair and its surroundings, unless this is a module relation.
    pub context: Option<PairContext<'a>>,
    /// The exterior degree of the pair, counting the generators of $C_0$ as exterior degree zero.
    pub ext_degree: u32,
    /// The number of reduction steps.
    pub steps: u64,
    /// The result.
    pub outcome: Outcome,
    /// The audited filters that predicted the pair to be useless.
    pub predicted_by: &'a [&'a str],
}

/// Receives an event for every reduction.
pub trait PairObserver: Send {
    /// Called for each reduction, in the order the results are processed.
    fn observe(&mut self, event: &PairEvent);
}

/// Writes every event as a line of JSON.
pub struct PairLog<W: Write + Send> {
    out: W,
}

impl<W: Write + Send> PairLog<W> {
    /// Log to `out`.
    pub fn new(out: W) -> Self {
        Self { out }
    }
}

impl<W: Write + Send> PairObserver for PairLog<W> {
    fn observe(&mut self, e: &PairEvent) {
        let mut line = format!(
            "{{\"s\":{},\"t\":{},\"ext\":{},\"steps\":{},\"outcome\":\"{}\"",
            e.s,
            e.t,
            e.ext_degree,
            e.steps,
            e.outcome.name()
        );
        match &e.context {
            None => line += ",\"kind\":\"relation\"",
            Some(ctx) => {
                let params = ctx.params;
                let lead =
                    |l: ModMono| format!("[{},\"{}\"]", l.generator(), params.format(l.mono()));
                let pair = ctx.pair;
                line += &format!(
                    ",\"kind\":\"{}\",\"i1\":{},\"i2\":{},\"m1\":\"{}\",\"m2\":\"{}\",\
                     \"lead1\":{},\"lead2\":{},\"lcm\":{},\"generator_degree\":{},\
                     \"relative_degree\":{},\"depth\":{}",
                    if pair.is_single() { "single" } else { "pair" },
                    if pair.is_single() {
                        "null".to_string()
                    } else {
                        pair.i1.to_string()
                    },
                    pair.i2,
                    params.format(pair.m1),
                    params.format(pair.m2),
                    ctx.lead1.map_or("null".to_string(), lead),
                    lead(ctx.lead2),
                    ctx.lcm()
                        .map_or("null".to_string(), |m| format!("\"{}\"", params.format(m))),
                    ctx.generator.degree,
                    ctx.relative_degree(),
                    ctx.depth(),
                );
            }
        }
        let predicted: Vec<String> = e.predicted_by.iter().map(|n| format!("\"{n}\"")).collect();
        line += &format!(",\"predicted_by\":[{}]}}", predicted.join(","));
        let _ = writeln!(self.out, "{line}");
    }
}

/// Counters for one level and degree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    /// How the critical pairs of this degree were created and pruned.
    pub pairs: PairCounts,
    /// The number of pairs, singles and relations that were reduced.
    pub reduced: u64,
    /// The number of reduction steps.
    pub steps: u64,
    /// Reductions with [`Outcome::Useless`].
    pub useless: u64,
    /// Reductions with [`Outcome::NewElement`].
    pub new_elements: u64,
    /// Reductions with [`Outcome::Boundary`].
    pub boundaries: u64,
    /// Reductions with [`Outcome::NewGenerator`].
    pub new_generators: u64,
    /// Reductions with [`Outcome::Beyond`].
    pub beyond: u64,
}

impl StepStats {
    /// Count an outcome.
    pub fn record(&mut self, outcome: Outcome, steps: u64) {
        self.reduced += 1;
        self.steps += steps;
        match outcome {
            Outcome::Useless => self.useless += 1,
            Outcome::NewElement => self.new_elements += 1,
            Outcome::Boundary => self.boundaries += 1,
            Outcome::NewGenerator => self.new_generators += 1,
            Outcome::Beyond => self.beyond += 1,
        }
    }

    /// Add the counters of `other`.
    fn add(&mut self, other: &Self) {
        self.pairs.formed += other.pairs.formed;
        self.pairs.divisible += other.pairs.divisible;
        self.pairs.chain += other.pairs.chain;
        self.pairs.singles += other.pairs.singles;
        self.reduced += other.reduced;
        self.steps += other.steps;
        self.useless += other.useless;
        self.new_elements += other.new_elements;
        self.boundaries += other.boundaries;
        self.new_generators += other.new_generators;
        self.beyond += other.beyond;
    }
}

/// Counters for every level and degree.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    steps: BTreeMap<(i32, i32), StepStats>,
}

impl Stats {
    /// The counters of level `s` in degree `t`.
    pub fn get_mut(&mut self, s: i32, t: i32) -> &mut StepStats {
        self.steps.entry((s, t)).or_default()
    }

    /// The counters of every level and degree, keyed by `(s, t)`.
    pub fn steps(&self) -> &BTreeMap<(i32, i32), StepStats> {
        &self.steps
    }

    /// The sums over all degrees, by level.
    pub fn by_level(&self) -> BTreeMap<i32, StepStats> {
        let mut out: BTreeMap<i32, StepStats> = BTreeMap::new();
        for (&(s, _), st) in &self.steps {
            out.entry(s).or_default().add(st);
        }
        out
    }

    /// The sum over everything.
    pub fn total(&self) -> StepStats {
        let mut total = StepStats::default();
        for st in self.steps.values() {
            total.add(st);
        }
        total
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
            "s",
            "formed",
            "divisible",
            "chain",
            "singles",
            "reduced",
            "useless",
            "elements",
            "boundary",
            "gens",
            "beyond",
            "steps"
        )?;
        let row = |f: &mut fmt::Formatter<'_>, label: String, st: &StepStats| {
            writeln!(
                f,
                "{label:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
                st.pairs.formed,
                st.pairs.divisible,
                st.pairs.chain,
                st.pairs.singles,
                st.reduced,
                st.useless,
                st.new_elements,
                st.boundaries,
                st.new_generators,
                st.beyond,
                st.steps
            )
        };
        for (s, st) in self.by_level() {
            row(f, s.to_string(), &st)?;
        }
        row(f, "all".to_string(), &self.total())
    }
}
