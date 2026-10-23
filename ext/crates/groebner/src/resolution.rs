//! The minimal resolution, computed one internal degree at a time.

use std::{collections::BTreeMap, sync::Arc};

use maybe_rayon::prelude::*;

use crate::{
    ModMono, Mono, Params, Poly,
    filter::{FilterMode, FilterReport, InstalledFilter, PairContext, PairFilter},
    level::{GbElement, Level},
    pairs::Pair,
    stats::{Outcome, PairEvent, PairObserver, Stats},
};

/// A generator of a free module in the resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generator {
    /// The internal degree.
    pub degree: i32,
    /// The exterior degree, counting the generators of $C_0$ as exterior degree zero.
    pub ext_degree: u32,
}

/// Which part of the resolution to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    /// The largest homological degree.
    pub s_max: u32,
    /// The largest stem $t - s$.
    pub n_max: i32,
}

/// The outcome of reducing one pair or relation, before it is sorted into the basis.
struct Reduced {
    s: i32,
    pair: Option<Pair>,
    predicted: Vec<u16>,
    element: GbElement,
    steps: u64,
}

/// A minimal free resolution $\dots \to C_1 \to C_0 \to M$ of a module $M$ given by generators and
/// relations, following Lin's algorithm.
///
/// Level `s` holds a Gröbner basis of the graph of $d \colon C_{s+1} \to C_s$. In each internal
/// degree, the critical pairs of every level are reduced. A pair whose $C_s$ component survives is
/// a new element of the basis. Otherwise its $C_{s+1}$ component is a cycle, which either is a
/// boundary or becomes a new generator of $C_{s+2}$.
pub struct Resolution {
    params: Arc<Params>,
    bounds: Bounds,
    generators: Vec<Vec<Generator>>,
    differentials: Vec<Vec<Poly>>,
    levels: Vec<Level>,
    relations: BTreeMap<i32, Vec<Poly>>,
    min_degree: i32,
    t_done: i32,
    t_target: i32,
    stats: Stats,
    observer: Option<Box<dyn PairObserver>>,
    filters: Vec<InstalledFilter>,
}

impl Resolution {
    /// A resolution of the module with generators in degrees `generator_degrees` (in increasing
    /// order) and the given relations.
    pub fn new(
        params: Arc<Params>,
        generator_degrees: &[i32],
        relations: Vec<Poly>,
        bounds: Bounds,
    ) -> Self {
        assert!(
            generator_degrees.is_sorted(),
            "generators must come in increasing degree"
        );
        let min_degree = generator_degrees.first().copied().unwrap_or(0);
        let mut by_degree: BTreeMap<i32, Vec<Poly>> = BTreeMap::new();
        for rel in relations {
            let Some((lead, _)) = rel.lead() else {
                continue;
            };
            let t = params.degree(lead.mono) + generator_degrees[lead.generator as usize];
            by_degree.entry(t).or_default().push(rel);
        }
        let mut generators = vec![Vec::new(); bounds.s_max as usize + 1];
        generators[0] = generator_degrees
            .iter()
            .map(|&degree| Generator {
                degree,
                ext_degree: 0,
            })
            .collect();
        Self {
            levels: (0..bounds.s_max)
                .map(|_| Level::new(min_degree - 1))
                .collect(),
            differentials: vec![Vec::new(); bounds.s_max as usize + 1],
            params,
            bounds,
            generators,
            relations: by_degree,
            min_degree,
            t_done: min_degree,
            t_target: min_degree,
            stats: Stats::default(),
            observer: None,
            filters: Vec::new(),
        }
    }

    /// A resolution of the ground field.
    pub fn sphere(params: Arc<Params>, bounds: Bounds) -> Self {
        let mut relations = Vec::new();
        let unit = |m: Mono| Poly::monomial(ModMono::new(0, m), 1);
        if params.has_exterior() {
            relations.push(unit(params.from_digits(1, [])));
        }
        for exp in 0.. {
            let Some(idx) = params.position_of(1, exp) else {
                break;
            };
            relations.push(unit(params.from_digits(0, [(idx, 1)])));
        }
        Self::new(params, &[0], relations, bounds)
    }

    /// Install an observer that sees every reduction.
    pub fn set_observer(&mut self, observer: Option<Box<dyn PairObserver>>) {
        self.observer = observer;
    }

    /// Install a filter, which applies to the pairs of every degree not computed yet.
    pub fn add_filter(&mut self, filter: Box<dyn PairFilter>, mode: FilterMode) {
        self.filters.push(InstalledFilter {
            filter,
            mode,
            counts: BTreeMap::new(),
        });
    }

    /// How the installed filters fared.
    pub fn filter_report(&self) -> FilterReport {
        FilterReport {
            filters: self
                .filters
                .iter()
                .map(|f| (f.filter.name().to_string(), f.mode, f.counts.clone()))
                .collect(),
        }
    }

    /// The algebra.
    pub fn params(&self) -> &Arc<Params> {
        &self.params
    }

    /// The bounds of the computation.
    pub fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// The degree of the bottom generator.
    pub fn min_degree(&self) -> i32 {
        self.min_degree
    }

    /// All internal degrees up to this one are done.
    pub fn t_done(&self) -> i32 {
        self.t_done
    }

    /// The generators of $C_s$, in order of increasing degree.
    pub fn generators(&self, s: u32) -> &[Generator] {
        &self.generators[s as usize]
    }

    /// The differential of the `i`th generator of $C_s$, an element of $C_{s-1}$.
    pub fn differential(&self, s: u32, i: usize) -> &Poly {
        &self.differentials[s as usize][i]
    }

    /// The counters collected so far.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The Gröbner basis of level `s`.
    pub fn level(&self, s: u32) -> &Level {
        &self.levels[s as usize]
    }

    /// The largest degree of a pair of level `s` that can matter.
    fn pair_t_trunc(&self, s: i32) -> i32 {
        self.t_target.min(self.bounds.n_max + s + 2)
    }

    /// Whether level `s` is needed in degree `t`. Its kernel there gives generators of $C_{s+2}$
    /// in stem $t - s - 2$.
    fn active(&self, s: i32, t: i32) -> bool {
        t <= self.bounds.n_max + s + 2
    }

    /// Compute everything in internal degrees up to `t_max`.
    pub fn compute_through(&mut self, t_max: i32) {
        if t_max > self.t_target {
            self.t_target = t_max;
            for s in 0..self.levels.len() {
                let t_trunc = self.pair_t_trunc(s as i32);
                if t_trunc > self.levels[s].t_trunc() {
                    let gens = &self.generators[s];
                    self.levels[s].retruncate(
                        &self.params,
                        |g| gens[g as usize].degree,
                        t_trunc,
                        self.t_done + 1,
                    );
                }
            }
        }
        while self.t_done < t_max {
            self.step(self.t_done + 1);
            self.t_done += 1;
        }
    }

    /// The context of a pair of level `s` in degree `t`.
    fn context<'a>(
        params: &'a Params,
        levels: &[Level],
        generators: &[Vec<Generator>],
        s: usize,
        t: i32,
        pair: &'a Pair,
    ) -> PairContext<'a> {
        let leads = levels[s].leads();
        let lead2 = leads[pair.i2 as usize];
        PairContext {
            params,
            s: s as u32,
            t,
            pair,
            lead1: (!pair.is_single()).then(|| leads[pair.i1 as usize]),
            lead2,
            generator: generators[s][lead2.generator as usize],
        }
    }

    /// Reduce the pairs and relations of degree `t` and sort the results into the bases.
    fn step(&mut self, t: i32) {
        let _span = tracing::info_span!("groebner_step", t).entered();
        let params = Arc::clone(&self.params);
        params.clear_product_cache();
        let p = params.prime();
        let num_levels = self.levels.len();

        // `work[s + 1]` are the pairs of level `s`, or the relations for `s = -1`, each with the
        // audited filters that predict it to be useless.
        let mut work: Vec<Vec<(Option<Pair>, Vec<u16>)>> = vec![Vec::new(); num_levels + 1];
        let relations = if self.active(-1, t) {
            self.relations.remove(&t).unwrap_or_default()
        } else {
            Vec::new()
        };
        work[0] = vec![(None, Vec::new()); relations.len()];
        for s in 0..num_levels {
            if !self.active(s as i32, t) {
                continue;
            }
            let (pairs, counts) = self.levels[s].take_pairs(&params, t);
            self.stats.get_mut(s as i32, t).pairs = counts;
            for pair in pairs {
                let ctx = Self::context(&params, &self.levels, &self.generators, s, t, &pair);
                let mut predicted = Vec::new();
                let mut discard = false;
                for (i, installed) in self.filters.iter_mut().enumerate() {
                    if !installed.filter.skip(&ctx) {
                        continue;
                    }
                    match installed.mode {
                        FilterMode::Enforce => {
                            installed.counts.entry(s as u32).or_default().discarded += 1;
                            discard = true;
                            break;
                        }
                        FilterMode::Audit => predicted.push(i as u16),
                    }
                }
                if !discard {
                    work[s + 1].push((Some(pair), predicted));
                }
            }
        }
        if work.iter().all(Vec::is_empty) {
            return;
        }

        // Each level is reduced by one task, so that the products cached while reducing one pair
        // are at hand for the others.
        let levels = &self.levels;
        let mut by_level: Vec<Vec<Reduced>> = work
            .into_maybe_par_iter()
            .enumerate()
            .map(|(level_idx, pairs)| {
                let s = level_idx as i32 - 1;
                pairs
                    .into_iter()
                    .enumerate()
                    .map(|(idx, (pair, predicted))| {
                        let (mut element, mut steps) = match pair {
                            None => (
                                GbElement {
                                    x1: Poly::zero(),
                                    x2: relations[idx].clone(),
                                },
                                0,
                            ),
                            Some(pair) => {
                                let level = &levels[s as usize];
                                let mut e = level.pair_element(&params, &pair);
                                let steps = level.reduce(&params, &mut e.x1, Some(&mut e.x2));
                                (e, steps)
                            }
                        };
                        if let Some(next) = levels.get((s + 1) as usize) {
                            steps += next.reduce(&params, &mut element.x2, None);
                        }
                        Reduced {
                            s,
                            pair,
                            predicted,
                            element,
                            steps,
                        }
                    })
                    .collect()
            })
            .collect();

        let mut new: Vec<Vec<GbElement>> = vec![Vec::new(); num_levels];
        let mut events: Vec<(Reduced, Outcome)> = Vec::new();
        for s in (-1..num_levels as i32).rev() {
            let mut kernels: Vec<Reduced> = Vec::new();
            for mut r in std::mem::take(&mut by_level[(s + 1) as usize]) {
                if s >= 0 {
                    Level::triangulate(
                        &params,
                        &mut r.element.x1,
                        Some(&mut r.element.x2),
                        &new[s as usize],
                    );
                }
                if let Some(next) = new.get((s + 1) as usize) {
                    Level::triangulate(&params, &mut r.element.x2, None, next);
                }
                if !r.element.x1.is_zero() {
                    r.element.normalize(&params);
                    new[s as usize].push(r.element.clone());
                    events.push((r, Outcome::NewElement));
                } else if !r.element.x2.is_zero() {
                    kernels.push(r);
                } else {
                    events.push((r, Outcome::Useless));
                }
            }

            for mut r in kernels {
                let s2 = (s + 2) as usize;
                if let Some(next) = new.get((s + 1) as usize) {
                    Level::triangulate(&params, &mut r.element.x2, None, next);
                }
                let Some((lead, c)) = r.element.x2.lead() else {
                    events.push((r, Outcome::Boundary));
                    continue;
                };
                if s2 > self.bounds.s_max as usize {
                    events.push((r, Outcome::NewGenerator));
                    continue;
                }
                r.element.x2.scale(params.inverse(c), p);
                let cycle = r.element.x2.clone();
                let ext_degree = lead.mono.ext_degree()
                    + self.generators[s2 - 1][lead.generator as usize].ext_degree;
                let index = self.generators[s2].len() as u32;
                self.generators[s2].push(Generator {
                    degree: t,
                    ext_degree,
                });
                self.differentials[s2].push(cycle.clone());
                if let Some(next) = new.get_mut(s2 - 1) {
                    next.push(GbElement {
                        x1: cycle,
                        x2: Poly::monomial(ModMono::new(index, Mono::ONE), 1),
                    });
                }
                events.push((r, Outcome::NewGenerator));
            }
        }

        for (r, outcome) in &events {
            self.stats.get_mut(r.s, t).record(*outcome, r.steps);
            if r.pair.is_none() {
                continue;
            }
            let useless = *outcome == Outcome::Useless;
            for (i, installed) in self.filters.iter_mut().enumerate() {
                let counts = installed.counts.entry(r.s as u32).or_default();
                match (r.predicted.contains(&(i as u16)), useless) {
                    (true, true) => counts.correct += 1,
                    (true, false) => {
                        counts.wrong += 1;
                        tracing::warn!(
                            filter = installed.filter.name(),
                            s = r.s,
                            t,
                            ?outcome,
                            "a pair predicted to be useless was not"
                        );
                    }
                    (false, true) => counts.missed += 1,
                    (false, false) => {}
                }
            }
        }
        if let Some(observer) = &mut self.observer {
            events.sort_by_key(|(r, _)| (r.s, r.pair.map(|p| (p.i2, p.i1))));
            for (r, outcome) in &events {
                let context = r.pair.as_ref().map(|pair| {
                    Self::context(
                        &params,
                        &self.levels,
                        &self.generators,
                        r.s as usize,
                        t,
                        pair,
                    )
                });
                let ext_degree = context.map_or(0, |ctx| {
                    ctx.pair.m2.ext_degree()
                        + ctx.lead2.mono.ext_degree()
                        + ctx.generator.ext_degree
                });
                let predicted_by: Vec<&str> = r
                    .predicted
                    .iter()
                    .map(|&i| self.filters[i as usize].filter.name())
                    .collect();
                observer.observe(&PairEvent {
                    s: r.s,
                    t,
                    context,
                    ext_degree,
                    steps: r.steps,
                    outcome: *outcome,
                    predicted_by: &predicted_by,
                });
            }
        }

        for (s, elements) in new.into_iter().enumerate() {
            for e in elements {
                let lead = e.x1.lead().unwrap().0;
                let degree = self.generators[s][lead.generator as usize].degree;
                self.levels[s].push(&params, e, degree);
            }
        }
        tracing::debug!(
            t,
            generators = ?(0..=self.bounds.s_max as usize)
                .map(|s| self.generators[s].iter().filter(|g| g.degree == t).count())
                .collect::<Vec<_>>(),
        );
    }
}
