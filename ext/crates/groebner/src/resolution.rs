//! The minimal resolution, computed one internal degree at a time.

use std::{collections::BTreeMap, sync::Arc};

use maybe_rayon::prelude::*;

use crate::{
    ModMono, Mono, Params, Poly,
    filter::{FilterCounts, FilterMode, FilterReport, InstalledFilter, PairContext, PairFilter},
    level::{GbElement, Level, X2mTrack, filtration},
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

/// The index standing for x2m pruning among the predictions of a pair.
const X2M_FILTER: u16 = u16::MAX;
/// The name of x2m pruning in reports and logs.
const X2M_NAME: &str = "x2m";

/// Lin's pruning of critical pairs through the associated graded algebra of the May filtration.
///
/// Every element of level `s` carries an x2m component, an element of a free module over the
/// associated graded algebra on the x2m generators of the level. A fresh generator is attached to
/// every element whose leading part is new: one coming from a cycle, and one whose leading part
/// changed filtration during reduction. When the leading part of a reduction cancels instead, its
/// x2m component is a relation among the leading parts of earlier elements. The relations are
/// collected into a Gröbner basis, and a critical pair whose x2m component reduces to zero modulo
/// the relations and the other pairs of its degree is predicted to be useless.
struct X2m {
    mode: FilterMode,
    /// The Gröbner bases of the relations, one per level.
    levels: Vec<Level>,
    /// `degrees[s][i]` is the degree of the `i`th x2m generator of level `s`.
    degrees: Vec<Vec<i32>>,
    counts: BTreeMap<u32, FilterCounts>,
}

impl X2m {
    /// A fresh x2m generator of level `s` in degree `t`.
    fn new_generator(&mut self, s: usize, t: i32) -> Poly {
        let index = self.degrees[s].len() as u32;
        self.degrees[s].push(t);
        Poly::monomial(ModMono::new(index, Mono::ONE), 1)
    }

    /// Add a relation with leading coefficient one to the basis of level `s`.
    fn push(&mut self, params: &Params, s: usize, rel: GbElement) {
        let lead = rel.x1.lead().unwrap().0;
        let degree = self.degrees[s][lead.generator as usize];
        self.levels[s].push(params, rel, degree);
    }

    /// Reduce the critical pairs of degree `t` of the relations of level `s`, and add the results
    /// to the basis.
    fn complete_basis(&mut self, params: &Params, s: usize, t: i32) {
        let (pairs, _) = self.levels[s].take_pairs(params, t);
        let mut new: Vec<GbElement> = Vec::new();
        for pair in pairs {
            let mut e = self.levels[s].pair_element(params, &pair, false);
            self.levels[s].reduce(params, &mut e.x1, None, None);
            Level::triangulate(params, &mut e.x1, None, &new);
            if !e.x1.is_zero() {
                e.normalize(params);
                new.push(e);
            }
        }
        for rel in new {
            self.push(params, s, rel);
        }
    }

    /// Mark the pairs of level `s` whose x2m components are spanned by the relations and the
    /// pairs before them, after ordering the pairs by decreasing filtration. In
    /// [`FilterMode::Enforce`] they are discarded instead.
    fn predict(
        &mut self,
        params: &Params,
        level: &Level,
        s: usize,
        mut pairs: Vec<(Option<Pair>, Vec<u16>)>,
    ) -> Vec<(Option<Pair>, Vec<u16>)> {
        let p = params.prime();
        let fil = |pair: &Option<Pair>| level.pair_filtration(params, pair.as_ref().unwrap());
        pairs.sort_by_key(|(pair, _)| std::cmp::Reverse(fil(pair)));
        let mut seen: Vec<GbElement> = Vec::new();
        let mut kept = Vec::with_capacity(pairs.len());
        for (pair, mut predicted) in pairs {
            let mut x = level.pair_x2m(params, pair.as_ref().unwrap());
            self.levels[s].reduce(params, &mut x, None, None);
            Level::triangulate(params, &mut x, None, &seen);
            if let Some((_, c)) = x.lead() {
                x.scale(params.inverse(c), p);
                seen.push(GbElement::new(x, Poly::zero()));
                kept.push((pair, predicted));
                continue;
            }
            match self.mode {
                FilterMode::Enforce => {
                    self.counts.entry(s as u32).or_default().discarded += 1;
                }
                FilterMode::Audit => {
                    predicted.push(X2M_FILTER);
                    kept.push((pair, predicted));
                }
            }
        }
        kept
    }
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
    x2m: Option<X2m>,
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
            x2m: None,
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

    /// Turn on Lin's x2m pruning. In [`FilterMode::Audit`], the pairs it would discard are
    /// reduced anyway and the predictions are recorded in the [`filter_report`](Self::filter_report)
    /// under the name "x2m".
    ///
    /// # Panics
    ///
    /// If anything has been computed already.
    pub fn enable_x2m(&mut self, mode: FilterMode) {
        assert_eq!(
            self.t_done, self.min_degree,
            "x2m pruning must be enabled before computing"
        );
        self.x2m = Some(X2m {
            mode,
            levels: (0..self.levels.len())
                .map(|_| Level::new_may(self.min_degree - 1))
                .collect(),
            degrees: vec![Vec::new(); self.levels.len()],
            counts: BTreeMap::new(),
        });
    }

    /// How the installed filters fared.
    pub fn filter_report(&self) -> FilterReport {
        let mut filters: Vec<_> = self
            .filters
            .iter()
            .map(|f| (f.filter.name().to_string(), f.mode, f.counts.clone()))
            .collect();
        if let Some(x2m) = &self.x2m {
            filters.push((X2M_NAME.to_string(), x2m.mode, x2m.counts.clone()));
        }
        FilterReport { filters }
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
                    if let Some(x2m) = &mut self.x2m {
                        let degrees = &x2m.degrees[s];
                        x2m.levels[s].retruncate(
                            &self.params,
                            |g| degrees[g as usize],
                            t_trunc,
                            self.t_done + 1,
                        );
                    }
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
        let track = self.x2m.is_some();

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
            if let Some(x2m) = &mut self.x2m {
                x2m.complete_basis(&params, s, t);
            }
            let (pairs, counts) = self.levels[s].take_pairs(&params, t);
            self.stats.get_mut(s as i32, t).pairs = counts;
            let mut kept = Vec::new();
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
                    kept.push((Some(pair), predicted));
                }
            }
            if let Some(x2m) = &mut self.x2m {
                kept = x2m.predict(&params, &self.levels[s], s, kept);
            }
            // A prediction is right if the pair adds nothing to the others, so reduce the pairs
            // under audit last. This changes which pairs become elements, but not their leading
            // terms.
            kept.sort_by_key(|(_, predicted)| !predicted.is_empty());
            work[s + 1] = kept;
        }
        if work.iter().all(Vec::is_empty) {
            return;
        }

        // Each level is reduced by one task, so that the products cached while reducing one pair
        // are at hand for the others.
        let levels = &self.levels;
        let x2m_levels = self.x2m.as_ref().map(|x| &x.levels);
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
                            None => (GbElement::new(Poly::zero(), relations[idx].clone()), 0),
                            Some(pair) => {
                                let level = &levels[s as usize];
                                let mut e = level.pair_element(&params, &pair, track);
                                let GbElement { x1, x2, x2m, fil } = &mut e;
                                let track = track.then_some(X2mTrack { x2m, fil: *fil });
                                let mut steps = level.reduce(&params, x1, Some(x2), track);
                                if let Some(x2m_levels) = x2m_levels {
                                    steps += x2m_levels[s as usize]
                                        .reduce(&params, &mut e.x2m, None, None);
                                }
                                (e, steps)
                            }
                        };
                        if let Some(next) = levels.get((s + 1) as usize) {
                            steps += next.reduce(&params, &mut element.x2, None, None);
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
        // New relations among the x2m generators of each level.
        let mut new_x2m: Vec<Vec<GbElement>> = vec![Vec::new(); num_levels];
        let mut events: Vec<(Reduced, Outcome)> = Vec::new();
        for s in (-1..num_levels as i32).rev() {
            let mut kernels: Vec<Reduced> = Vec::new();
            let mut x2m_cycles: Vec<Poly> = Vec::new();
            for mut r in std::mem::take(&mut by_level[(s + 1) as usize]) {
                if s >= 0 {
                    Level::triangulate_element(&params, &mut r.element, &new[s as usize], track);
                }
                if let Some(next) = new.get((s + 1) as usize) {
                    Level::triangulate(&params, &mut r.element.x2, None, next);
                }
                if let Some((lead, _)) = r.element.x1.lead() {
                    let valid = r.element.x2m_valid(&params);
                    r.element.normalize(&params);
                    if let Some(x2m) = &mut self.x2m
                        && !valid
                    {
                        x2m_cycles.push(std::mem::take(&mut r.element.x2m));
                        r.element.x2m = x2m.new_generator(s as usize, t);
                        r.element.fil = filtration(&params, lead);
                    }
                    new[s as usize].push(r.element.clone());
                    events.push((r, Outcome::NewElement));
                } else {
                    if track && !r.element.x2m.is_zero() {
                        x2m_cycles.push(std::mem::take(&mut r.element.x2m));
                    }
                    if r.element.x2.is_zero() {
                        events.push((r, Outcome::Useless));
                    } else {
                        kernels.push(r);
                    }
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
                    events.push((r, Outcome::Beyond));
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
                if new.get(s2 - 1).is_some() {
                    let mut element =
                        GbElement::new(cycle, Poly::monomial(ModMono::new(index, Mono::ONE), 1));
                    if let Some(x2m) = &mut self.x2m {
                        element.x2m = x2m.new_generator(s2 - 1, t);
                        element.fil = filtration(&params, lead);
                    }
                    new[s2 - 1].push(element);
                }
                events.push((r, Outcome::NewGenerator));
            }

            if s >= 0 {
                for mut x in x2m_cycles {
                    let rels = &new_x2m[s as usize];
                    Level::triangulate(&params, &mut x, None, rels);
                    if !x.is_zero() {
                        let mut rel = GbElement::new(x, Poly::zero());
                        rel.normalize(&params);
                        new_x2m[s as usize].push(rel);
                    }
                }
            }
        }

        for (r, outcome) in &events {
            self.stats.get_mut(r.s, t).record(*outcome, r.steps);
            if r.pair.is_none() {
                continue;
            }
            let useless = outcome.is_redundant();
            let record =
                |counts: &mut FilterCounts, predicted: bool, name: &str| match (predicted, useless)
                {
                    (true, true) => counts.correct += 1,
                    (true, false) => {
                        counts.wrong += 1;
                        tracing::warn!(
                            filter = name,
                            s = r.s,
                            t,
                            ?outcome,
                            "a pair predicted to be useless was not"
                        );
                    }
                    (false, true) => counts.missed += 1,
                    (false, false) => {}
                };
            for (i, installed) in self.filters.iter_mut().enumerate() {
                let counts = installed.counts.entry(r.s as u32).or_default();
                record(
                    counts,
                    r.predicted.contains(&(i as u16)),
                    installed.filter.name(),
                );
            }
            if let Some(x2m) = &mut self.x2m {
                let counts = x2m.counts.entry(r.s as u32).or_default();
                record(counts, r.predicted.contains(&X2M_FILTER), X2M_NAME);
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
                    .map(|&i| {
                        if i == X2M_FILTER {
                            X2M_NAME
                        } else {
                            self.filters[i as usize].filter.name()
                        }
                    })
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
        if let Some(x2m) = &mut self.x2m {
            for (s, rels) in new_x2m.into_iter().enumerate() {
                for rel in rels {
                    x2m.push(&params, s, rel);
                }
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
