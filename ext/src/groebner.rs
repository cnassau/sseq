//! Resolutions computed by Lin's Gröbner basis algorithm, as chain complexes.

use std::sync::{Arc, Mutex};

use ::groebner::{Bounds, ModMono, Mono, Params, Poly};
use algebra::{
    Algebra,
    milnor_algebra::{MilnorAlgebra, MilnorBasisElement, PPart},
    module::{
        FreeModule, Module, ZeroModule,
        homomorphism::{FreeModuleHomomorphism, FullModuleHomomorphism},
    },
};
use anyhow::anyhow;
use fp::{
    prime::{Prime, ValidPrime},
    vector::FpVector,
};
use once::OnceBiVec;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FiniteChainComplex},
    save::SaveDirectory,
};

/// A finite module presented by generators and relations over the free module on its generators.
struct Presentation {
    /// The degrees of the generators, in increasing order.
    generator_degrees: Vec<i32>,
    /// Each generator is the basis element with this index of the module in its degree.
    generator_images: Vec<usize>,
    relations: Vec<Poly>,
}

/// The index of `m` in the Milnor basis of `algebra`.
fn milnor_index(params: &Params, algebra: &MilnorAlgebra, m: Mono) -> usize {
    let (q_part, r) = params.decode(m);
    algebra.basis_element_to_index(&MilnorBasisElement {
        q_part,
        p_part: PPart::from_slice(&r),
        degree: params.degree(m),
    })
}

/// Present `module` by generators and enough relations to determine it through degree `t_max`.
///
/// Every basis element $b$ of the module gets a preimage $w_b$ in the free module, and the
/// relations are the elements $g w_b - w_{g b}$ for algebra generators $g$, which generate the
/// kernel. Finding the $w_b$ and the generators is linear algebra on the module, which is small.
fn presentation(
    params: &Params,
    algebra: &MilnorAlgebra,
    module: &impl Module<Algebra = MilnorAlgebra>,
    t_max: i32,
) -> Presentation {
    let p = params.prime();
    let min_degree = module.min_degree();
    let max_degree = module.max_degree().expect("module must be finite");

    let mut algebra_generators: Vec<(Mono, i32)> = Vec::new();
    if params.has_exterior() {
        let q0 = params.from_digits(1, []);
        algebra_generators.push((q0, params.degree(q0)));
    }
    for exp in 0.. {
        let Some(idx) = params.position_of(1, exp) else {
            break;
        };
        let m = params.from_digits(0, [(idx, 1)]);
        if params.degree(m) > t_max - min_degree {
            break;
        }
        algebra_generators.push((m, params.degree(m)));
    }
    algebra.compute_basis(t_max - min_degree);

    let mut presentation = Presentation {
        generator_degrees: Vec::new(),
        generator_images: Vec::new(),
        relations: Vec::new(),
    };
    // `preimages[t - min_degree][j]` is the chosen preimage of basis element `j` in degree `t`.
    let mut preimages: Vec<Vec<Poly>> = Vec::new();
    for t in min_degree..=t_max {
        let dim = if t <= max_degree {
            module.dimension(t)
        } else {
            0
        };
        // Rows (v, w) with v in the module and w a preimage of v.
        let mut rows: Vec<(Vec<u32>, Poly)> = Vec::new();
        for &(g, d) in &algebra_generators {
            let t0 = t - d;
            if t0 < min_degree || t0 > max_degree {
                continue;
            }
            let op_index = milnor_index(params, algebra, g);
            for (j, w) in preimages[(t0 - min_degree) as usize].iter().enumerate() {
                let mut v = FpVector::new(ValidPrime::new(p), dim);
                if dim > 0 {
                    module.act_on_basis(v.as_slice_mut(), 1, d, op_index, t0, j);
                }
                rows.push((v.iter().collect(), params.mul_poly(g, 1, w)));
            }
        }

        // Reduced row echelon form, with `pivot_rows[c]` the row whose pivot is column `c`.
        let mut pivot_rows: Vec<Option<(Vec<u32>, Poly)>> = vec![None; dim];
        let eliminate = |row: &mut (Vec<u32>, Poly), pivot: &(Vec<u32>, Poly), col: usize| {
            let c = row.0[col];
            if c != 0 {
                let f = (p - c) % p;
                for (x, y) in row.0.iter_mut().zip(&pivot.0) {
                    *x = (*x + f * y) % p;
                }
                row.1.add_scaled(&pivot.1, f, p);
            }
        };
        let install = |pivot_rows: &mut Vec<Option<(Vec<u32>, Poly)>>,
                       mut row: (Vec<u32>, Poly),
                       col: usize| {
            let inv = params.inverse(row.0[col]);
            for x in &mut row.0 {
                *x = *x * inv % p;
            }
            row.1.scale(inv, p);
            for other in pivot_rows.iter_mut().flatten() {
                eliminate(other, &row, col);
            }
            pivot_rows[col] = Some(row);
        };
        for mut row in rows {
            for (col, pivot) in pivot_rows.iter().enumerate() {
                if let Some(pivot) = pivot {
                    eliminate(&mut row, pivot, col);
                }
            }
            match row.0.iter().position(|&x| x != 0) {
                Some(col) => install(&mut pivot_rows, row, col),
                None => {
                    if !row.1.is_zero() {
                        presentation.relations.push(row.1);
                    }
                }
            }
        }
        for col in 0..dim {
            if pivot_rows[col].is_none() {
                let generator = presentation.generator_degrees.len() as u32;
                presentation.generator_degrees.push(t);
                presentation.generator_images.push(col);
                let mut v = vec![0; dim];
                v[col] = 1;
                let w = Poly::monomial(ModMono::new(generator, Mono::ONE), 1);
                install(&mut pivot_rows, (v, w), col);
            }
        }
        preimages.push(pivot_rows.into_iter().map(|r| r.unwrap().1).collect());
    }
    presentation
}

/// The inner resolution, together with how much of it has been copied into the free modules.
struct State {
    inner: Option<::groebner::Resolution>,
    /// `generator_images[i]` is the basis element of the module hit by generator `i` of $C_0$.
    generator_images: Vec<usize>,
}

/// A minimal resolution computed with [`groebner::Resolution`](::groebner::Resolution), wrapped in
/// the free modules and differentials the rest of the crate works with.
///
/// The whole range must be requested by one call to
/// [`compute_through_stem`](Self::compute_through_stem) or
/// [`compute_through_bidegree`](ChainComplex::compute_through_bidegree). A later call may only
/// ask for less, because the inner resolution only computes up to the bounds it was created with.
pub struct Resolution<M: ZeroModule<Algebra = MilnorAlgebra>> {
    lock: Mutex<State>,
    name: String,
    params: Arc<Params>,
    modules: OnceBiVec<Arc<FreeModule<MilnorAlgebra>>>,
    zero_module: Arc<FreeModule<MilnorAlgebra>>,
    differentials: OnceBiVec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>>,
    target: Arc<FiniteChainComplex<M>>,
    chain_maps: OnceBiVec<Arc<FreeModuleHomomorphism<M>>>,
    save_dir: SaveDirectory,
    observer: Mutex<Option<Box<dyn ::groebner::PairObserver>>>,
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> Resolution<M> {
    /// A resolution of `module`, which must be finite.
    pub fn new(module: Arc<M>) -> anyhow::Result<Self> {
        let algebra = module.algebra();
        if !algebra.profile().is_trivial() {
            return Err(anyhow!(
                "Gröbner resolutions need the full Steenrod algebra"
            ));
        }
        if module.max_degree().is_none() {
            return Err(anyhow!("Gröbner resolutions need a finite module"));
        }
        let params = Arc::new(Params::new(algebra.prime().as_u32(), algebra.kind()));
        let target = Arc::new(FiniteChainComplex::ccdz(module));
        Ok(Self {
            lock: Mutex::new(State {
                inner: None,
                generator_images: Vec::new(),
            }),
            name: String::new(),
            params,
            zero_module: Arc::new(FreeModule::new(algebra, "F_{-1}".to_string(), 0)),
            modules: OnceBiVec::new(0),
            differentials: OnceBiVec::new(0),
            chain_maps: OnceBiVec::new(0),
            target,
            save_dir: SaveDirectory::None,
            observer: Mutex::new(None),
        })
    }

    /// The name of the resolution.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the name of the resolution.
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Install an observer that sees every reduction of the next computation.
    pub fn set_observer(&self, observer: Option<Box<dyn ::groebner::PairObserver>>) {
        *self.observer.lock().unwrap() = observer;
    }

    /// The counters of the computation, if it has started.
    pub fn stats(&self) -> Option<::groebner::Stats> {
        let state = self.lock.lock().unwrap();
        state.inner.as_ref().map(|r| r.stats().clone())
    }

    /// The parameters of the algebra.
    pub fn params(&self) -> &Arc<Params> {
        &self.params
    }

    /// Create the modules, differentials and chain maps of homological degree up to `max_s`.
    fn extend_through_degree(&self, max_s: i32) {
        let min_degree = self.min_degree();
        self.modules.extend(max_s, |i| {
            Arc::new(FreeModule::new(self.algebra(), format!("F{i}"), min_degree))
        });
        self.differentials.extend(0, |_| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[0]),
                Arc::clone(&self.zero_module),
                0,
            ))
        });
        self.differentials.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                Arc::clone(&self.modules[i - 1]),
                0,
            ))
        });
        self.chain_maps.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                self.target.module(i),
                0,
            ))
        });
    }

    /// Compute everything in homological degree at most `max.s()` and stem at most `max.n()`.
    pub fn compute_through_stem(&self, max: Bidegree) {
        self.compute(max.s(), max.n(), |s| max.n() + s);
    }

    /// Compute the bidegrees `(s, t)` with `s <= max_s` and `t <= limit(s)`, which are all of stem
    /// at most `max_n`.
    fn compute(&self, max_s: i32, max_n: i32, limit: impl Fn(i32) -> i32) {
        let mut state = self.lock.lock().unwrap();
        let min_degree = self.min_degree();
        let bounds = Bounds {
            s_max: max_s as u32,
            n_max: max_n,
        };
        let t_max = (0..=max_s).map(&limit).max().unwrap();

        let fits = state
            .inner
            .as_ref()
            .is_some_and(|r| r.bounds().s_max >= bounds.s_max && r.bounds().n_max >= bounds.n_max);
        if !fits {
            assert!(
                state.inner.is_none(),
                "a Gröbner resolution cannot be extended beyond its first request"
            );
            let module = self.target.module(0);
            let algebra = self.algebra();
            let presentation = presentation(&self.params, &algebra, &*module, t_max);
            let mut inner = ::groebner::Resolution::new(
                Arc::clone(&self.params),
                &presentation.generator_degrees,
                presentation.relations,
                bounds,
            );
            inner.set_observer(self.observer.lock().unwrap().take());
            state.generator_images = presentation.generator_images;
            state.inner = Some(inner);
        }
        let inner = state.inner.as_mut().unwrap();
        inner.compute_through(t_max);

        self.extend_through_degree(max_s);
        self.algebra().compute_basis(t_max + 1 - min_degree);
        self.commit(&state, max_s, limit);
    }

    /// Copy the generators and differentials of the bidegrees `(s, t)` with `s <= max_s` and
    /// `t <= limit(s)` into the free modules.
    fn commit(&self, state: &State, max_s: i32, limit: impl Fn(i32) -> i32) {
        let inner = state.inner.as_ref().unwrap();
        let algebra = self.algebra();
        let p = self.prime();

        // `(degree, index within degree)` of each generator of C_s.
        let positions = |s: i32| -> Vec<(i32, usize)> {
            let mut out = Vec::new();
            let mut last = (i32::MIN, 0);
            for g in inner.generators(s as u32) {
                last = if g.degree == last.0 {
                    (g.degree, last.1 + 1)
                } else {
                    (g.degree, 0)
                };
                out.push(last);
            }
            out
        };

        for s in 0..=max_s {
            let module = &self.modules[s];
            let gens = inner.generators(s as u32);
            for t in module.gen_names().len()..=limit(s) {
                let num = gens.iter().filter(|g| g.degree == t).count();
                let names = (0..num)
                    .map(|idx| format!("x_{:#}", BidegreeGenerator::new(Bidegree::s_t(s, t), idx)))
                    .collect();
                module.add_generators(t, num, Some(names));
            }
            // The differential out of degree `limit(s + 1)` only hits generators of lower degree,
            // which may lie beyond `limit(s)`.
            module.compute_basis(limit(s).max(limit(s + 1)));
        }
        self.zero_module.extend_by_zero(limit(0));

        for s in 0..=max_s {
            let d = &self.differentials[s];
            let chain_map = &self.chain_maps[s];
            let gens = inner.generators(s as u32);
            let target_positions = if s > 0 { positions(s - 1) } else { Vec::new() };
            let first = gens.partition_point(|g| g.degree < d.next_degree());
            let mut i = first;
            for t in d.next_degree()..=limit(s) {
                let start = i;
                while i < gens.len() && gens[i].degree == t {
                    i += 1;
                }
                if s == 0 {
                    d.extend_by_zero(t);
                    let target = self.target.module(0);
                    let dim = target.dimension(t);
                    let rows = (start..i)
                        .map(|g| {
                            let mut v = FpVector::new(p, dim);
                            v.set_entry(state.generator_images[g], 1);
                            v
                        })
                        .collect();
                    chain_map.add_generators_from_rows(t, rows);
                    continue;
                }
                let target = &self.modules[s - 1];
                let dim = target.dimension_from_gens_below(t, t);
                let rows = (start..i)
                    .map(|g| {
                        let mut v = FpVector::new(p, dim);
                        for &(m, c) in inner.differential(s as u32, g).terms() {
                            let (gen_deg, gen_idx) = target_positions[m.generator as usize];
                            let op_deg = self.params.degree(m.mono);
                            let op_idx = milnor_index(&self.params, &algebra, m.mono);
                            debug_assert_eq!(op_deg + gen_deg, t);
                            let idx = target.generator_offset(t, gen_deg, gen_idx) + op_idx;
                            v.add_basis_element(idx, c);
                        }
                        v
                    })
                    .collect();
                d.add_generators_from_rows(t, rows);
                chain_map.extend_by_zero(t);
            }
        }
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> ChainComplex for Resolution<M> {
    type Algebra = MilnorAlgebra;
    type Homomorphism = FreeModuleHomomorphism<FreeModule<Self::Algebra>>;
    type Module = FreeModule<Self::Algebra>;

    fn algebra(&self) -> Arc<Self::Algebra> {
        self.zero_module.algebra()
    }

    fn module(&self, s: i32) -> Arc<Self::Module> {
        Arc::clone(&self.modules[s])
    }

    fn zero_module(&self) -> Arc<Self::Module> {
        Arc::clone(&self.zero_module)
    }

    fn min_degree(&self) -> i32 {
        self.target.min_degree()
    }

    fn has_computed_bidegree(&self, b: Bidegree) -> bool {
        self.differentials.len() > b.s() && self.differential(b.s()).next_degree() > b.t()
    }

    fn differential(&self, s: i32) -> Arc<Self::Homomorphism> {
        Arc::clone(&self.differentials[s])
    }

    fn compute_through_bidegree(&self, max: Bidegree) {
        // Every bidegree of the range has stem at most `max.t()`.
        self.compute(max.s(), max.t(), |_| max.t());
    }

    fn next_homological_degree(&self) -> i32 {
        self.modules.len()
    }

    fn save_dir(&self) -> &SaveDirectory {
        &self.save_dir
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> AugmentedChainComplex for Resolution<M> {
    type ChainMap = FreeModuleHomomorphism<M>;
    type TargetComplex = FiniteChainComplex<M, FullModuleHomomorphism<M, M>>;

    fn target(&self) -> Arc<Self::TargetComplex> {
        Arc::clone(&self.target)
    }

    fn chain_map(&self, s: i32) -> Arc<Self::ChainMap> {
        Arc::clone(&self.chain_maps[s])
    }
}
