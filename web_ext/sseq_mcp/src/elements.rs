//! Elements of the free modules `C_s` as text, and lifting cycles through the differential.

use std::collections::BTreeMap;

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart, PPartEntry},
    module::{Module, homomorphism::ModuleHomomorphism},
};
use ext::chain_complex::{ChainComplex, FreeChainComplex};
use fp::{prime::Prime, vector::FpVector};
use sseq::coordinates::Bidegree;

use crate::Res;

/// The name `x_s,t`, or `x_s,t[i]` when `C_s` has several generators in degree `t`.
pub fn generator_name(res: &Res, s: i32, t: i32, i: usize) -> String {
    if res.number_of_gens_in_bidegree(Bidegree::s_t(s, t)) > 1 {
        format!("x_{s},{t}[{i}]")
    } else {
        format!("x_{s},{t}")
    }
}

/// `v`, an element of `C_s` in internal degree `t`, as a sum of Milnor basis elements times
/// generators: `Sq(r_1, r_2, ...)` at p = 2, `Q_i ... P(r_1, ...)` with coefficients otherwise.
///
/// [`parse`] reads this format back.
pub fn element_string(res: &Res, s: i32, t: i32, v: &FpVector) -> String {
    let algebra = res.algebra();
    let algebra: &MilnorAlgebra = (&*algebra)
        .try_into()
        .expect("resolutions use the Milnor basis");
    let generic = algebra.prime() != 2;
    let module = res.module(s);
    let terms: Vec<String> = v
        .iter_nonzero()
        .map(|(index, c)| {
            let opgen = module.index_to_op_gen(t, index);
            let elt =
                algebra.basis_element_from_index(opgen.operation_degree, opgen.operation_index);
            let op = if elt.degree == 0 {
                "1".to_string()
            } else if generic {
                elt.to_string()
            } else {
                let r: Vec<String> = elt.p_part.iter().map(|r| r.to_string()).collect();
                format!("Sq({})", r.join(","))
            };
            let c = if c == 1 {
                String::new()
            } else {
                format!("{c} ")
            };
            let g = generator_name(res, s, opgen.generator_degree, opgen.generator_index);
            format!("{c}{op} {g}")
        })
        .collect();
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// One term `c · op · x_s,t[i]` of an element, before it is checked against a resolution.
#[derive(Debug, PartialEq)]
struct Term {
    coeff: u32,
    q_part: u32,
    p_part: Vec<PPartEntry>,
    s: i32,
    gen_t: i32,
    gen_idx: usize,
}

/// An element of `C_s` in internal degree `t`, as parsed by [`parse`].
pub struct Element {
    pub s: i32,
    pub t: i32,
    terms: Vec<(u32, MilnorBasisElement, i32, usize)>,
}

/// Parse an element in the format of [`element_string`], such as `Sq(2) x_1,2 + Sq(3) x_1,1` or
/// `2 Q_0 P(1) x_1,4`. `Sq(...)` and `P(...)` both give the P-part. The literal `0` is not accepted,
/// since it has no bidegree.
///
/// This checks the syntax and that all terms lie in one bidegree; [`Element::vector`] checks the
/// terms against the resolution once it is computed that far.
pub fn parse(res: &Res, text: &str) -> anyhow::Result<Element> {
    let algebra = res.algebra();
    let algebra: &MilnorAlgebra = (&*algebra).try_into()?;
    let p = algebra.prime();
    let mut terms = Vec::new();
    let mut bidegree = None;
    for term in text.split('+') {
        let term = parse_term(term.trim())
            .map_err(|e| anyhow::anyhow!("cannot read the term {:?}: {e}", term.trim()))?;
        anyhow::ensure!(
            term.q_part == 0 || algebra.has_exterior(),
            "Q_i are not basis elements here; write Sq(...) at p = 2"
        );
        let p_part = PPart::try_from_slice(&term.p_part)
            .ok_or_else(|| anyhow::anyhow!("the operation in {term:?} is too large"))?;
        let mut elt = MilnorBasisElement {
            q_part: term.q_part,
            p_part,
            degree: 0,
        };
        algebra.compute_degree(&mut elt);
        let b = Bidegree::s_t(term.s, term.gen_t + elt.degree);
        match bidegree {
            None => bidegree = Some(b),
            Some(first) => anyhow::ensure!(
                first == b,
                "terms lie in different bidegrees (s, t) = ({}, {}) and ({}, {})",
                first.s(),
                first.t(),
                b.s(),
                b.t()
            ),
        }
        terms.push((term.coeff % p.as_u32(), elt, term.gen_t, term.gen_idx));
    }
    let b = bidegree.unwrap();
    Ok(Element {
        s: b.s(),
        t: b.t(),
        terms,
    })
}

impl Element {
    /// The element as a vector in `C_s`, which must be computed through degree `t`.
    pub fn vector(&self, res: &Res) -> anyhow::Result<FpVector> {
        let algebra = res.algebra();
        let algebra: &MilnorAlgebra = (&*algebra).try_into()?;
        let module = res.module(self.s);
        let mut v = FpVector::new(res.prime(), module.dimension(self.t));
        for &(c, ref elt, gen_t, gen_idx) in &self.terms {
            let gens = module.number_of_gens_in_degree(gen_t);
            anyhow::ensure!(
                gen_idx < gens,
                "C_{} has {gens} generators in degree {gen_t}, so no x_{},{gen_t}[{gen_idx}]",
                self.s,
                self.s
            );
            let op_idx = algebra.try_basis_element_to_index(elt).ok_or_else(|| {
                anyhow::anyhow!("{elt} is not in the algebra this resolution is over")
            })?;
            let index = module.operation_generator_to_index(elt.degree, op_idx, gen_t, gen_idx);
            v.add_basis_element(index, c);
        }
        Ok(v)
    }
}

/// Parse `[c] [op] x_s,t[[i]]`, where `op` is `1`, `Q_i`s and an optional `Sq(...)` or `P(...)`.
fn parse_term(term: &str) -> Result<Term, String> {
    let at = term.rfind("x_").ok_or("no generator x_s,t")?;
    let (prefix, generator) = term.split_at(at);
    let (s, gen_t, gen_idx) = parse_generator(&generator[2..])?;

    let mut rest = prefix.trim().trim_end_matches('*').trim();
    let mut coeff = 1;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits > 0 && (digits == rest.len() || rest[digits..].starts_with([' ', '*'])) {
        coeff = rest[..digits].parse().map_err(|_| "bad coefficient")?;
        rest = rest[digits..].trim_start_matches([' ', '*']);
    }

    let mut q_part = 0;
    let mut p_part = Vec::new();
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("Q_") {
            let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            let i: u32 = r[..end].parse().map_err(|_| "bad Q_i")?;
            if i >= u32::BITS {
                return Err("Q_i index too large".into());
            }
            q_part |= 1 << i;
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix("Sq(").or_else(|| rest.strip_prefix("P(")) {
            let end = r.find(')').ok_or("unclosed parenthesis")?;
            if !p_part.is_empty() {
                return Err("two P-parts".into());
            }
            p_part = r[..end]
                .split(',')
                .map(|x| x.trim().parse::<PPartEntry>())
                .collect::<Result<_, _>>()
                .map_err(|_| "bad exponent")?;
            rest = &r[end + 1..];
        } else if let Some(r) = rest.strip_prefix('1') {
            rest = r;
        } else {
            return Err(format!("cannot read {rest:?}"));
        }
        rest = rest.trim_start_matches([' ', '*']);
    }
    Ok(Term {
        coeff,
        q_part,
        p_part,
        s,
        gen_t,
        gen_idx,
    })
}

/// Parse `s,t` or `s,t[i]`, the part of a generator name after `x_`.
fn parse_generator(text: &str) -> Result<(i32, i32, usize), String> {
    let (st, idx) = match text.split_once('[') {
        Some((st, idx)) => {
            let idx = idx.strip_suffix(']').ok_or("unclosed [")?;
            (st, idx.parse().map_err(|_| "bad generator index")?)
        }
        None => (text, 0),
    };
    let (s, t) = st.split_once(',').ok_or("a generator is x_s,t")?;
    let s = s.trim().parse().map_err(|_| "bad s")?;
    let t = t.trim().parse().map_err(|_| "bad t")?;
    Ok((s, t, idx))
}

/// Solve `d(y) = z` for each cycle `z`, lifting all cycles of one bidegree in one application of
/// the quasi-inverse. A cycle that is not a boundary gets an explanation instead.
///
/// `res` must be computed through `(s + 1, t)` for every cycle in `C_s` of degree `t`.
pub fn lift(res: &Res, cycles: &[(i32, i32, FpVector)]) -> Vec<Result<FpVector, String>> {
    let mut by_bidegree: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    for (k, (s, t, _)) in cycles.iter().enumerate() {
        by_bidegree.entry((*s, *t)).or_default().push(k);
    }
    let p = res.prime();
    let mut lifts: Vec<Option<Result<FpVector, String>>> = vec![None; cycles.len()];
    for ((s, t), ks) in by_bidegree {
        let dim = res.module(s + 1).dimension(t);
        let inputs: Vec<&FpVector> = ks.iter().map(|&k| &cycles[k].2).collect();
        let mut results: Vec<FpVector> = ks.iter().map(|_| FpVector::new(p, dim)).collect();
        if dim > 0 {
            let mut slices: Vec<_> = results.iter_mut().map(|r| r.as_slice_mut()).collect();
            let inputs: Vec<_> = inputs.iter().map(|z| z.as_slice()).collect();
            if !res.apply_quasi_inverse(&mut slices, Bidegree::s_t(s + 1, t), &inputs) {
                for &k in &ks {
                    lifts[k] = Some(Err(format!(
                        "no quasi-inverse is stored for (s, t) = ({}, {t})",
                        s + 1
                    )));
                }
                continue;
            }
        }
        for (&k, y) in ks.iter().zip(results) {
            let z = &cycles[k].2;
            let mut dy = FpVector::new(p, z.len());
            res.differential(s + 1)
                .apply(dy.as_slice_mut(), 1, t, y.as_slice());
            lifts[k] = Some(if dy == *z {
                Ok(y)
            } else {
                Err(not_a_boundary(res, s, t, z))
            });
        }
    }
    lifts.into_iter().map(Option::unwrap).collect()
}

/// Why `z`, an element of `C_s` in degree `t` without a lift, is not a boundary.
fn not_a_boundary(res: &Res, s: i32, t: i32, z: &FpVector) -> String {
    if s == 0 {
        return "not a boundary: its image under the augmentation C_0 -> M is nonzero".to_string();
    }
    let mut dz = FpVector::new(res.prime(), res.module(s - 1).dimension(t));
    res.differential(s)
        .apply(dz.as_slice_mut(), 1, t, z.as_slice());
    format!("not a cycle: d(z) = {}", element_string(res, s - 1, t, &dz))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms() {
        let term = |s| parse_term(s).unwrap();
        assert_eq!(
            term("Sq(6,1) x_1,8"),
            Term {
                coeff: 1,
                q_part: 0,
                p_part: vec![6, 1],
                s: 1,
                gen_t: 8,
                gen_idx: 0
            }
        );
        assert_eq!(term("1 x_0,0").p_part, Vec::<PPartEntry>::new());
        assert_eq!(term("x_5,20[1]").gen_idx, 1);
        let odd = term("2 Q_0 Q_1 P(1, 2) x_3,14");
        assert_eq!((odd.coeff, odd.q_part, odd.p_part), (2, 0b11, vec![1, 2]));
        assert_eq!(term("Sq(2)*x_1,2").p_part, vec![2]);
        for bad in [
            "Sq(2)",
            "Sq(2 x_1,1",
            "Foo x_1,1",
            "x_1",
            "x_1,1[",
            "Q_99 x_1,1",
        ] {
            assert!(parse_term(bad).is_err(), "{bad}");
        }
    }

    /// Lifting, and the explanations for cycles that have no lift.
    #[test]
    fn lifts() {
        let res = ext::utils::construct(("S_2", "milnor"), None).unwrap();
        res.compute_through_stem(Bidegree::n_s(16, 6));
        let element = |text| {
            let e = parse(&res, text).unwrap();
            (e.s, e.t, e.vector(&res).unwrap())
        };
        let cycles = [
            // d(x_1,1) = Sq(1) x_0,0, so Sq(1) x_0,0 lifts to x_1,1.
            element("Sq(1) x_0,0"),
            // Sq(1) Sq(1) = 0, so Sq(1) x_1,1 is a cycle in C_1 and lifts to x_2,2.
            element("Sq(1) x_1,1"),
            // A cycle since Sq(2) Sq(2) = Sq(3) Sq(1); it lifts to h1^2 up to h0 multiples.
            element("Sq(2) x_1,2 + Sq(3) x_1,1"),
            // The unit generator survives the augmentation.
            element("1 x_0,0"),
            // d(Sq(1) x_1,2) = Sq(1) Sq(2) x_0,0 = Sq(3) x_0,0, which is not zero.
            element("Sq(1) x_1,2"),
        ];
        let lifts = lift(&res, &cycles);
        let shown: Vec<String> = lifts
            .iter()
            .zip(&cycles)
            .map(|(l, (s, t, _))| match l {
                Ok(y) => element_string(&res, s + 1, *t, y),
                Err(e) => e.clone(),
            })
            .collect();
        assert_eq!(shown[0], "1 x_1,1");
        assert_eq!(shown[1], "1 x_2,2");
        assert!(lifts[2].is_ok(), "{}", shown[2]);
        assert!(shown[3].contains("augmentation"), "{}", shown[3]);
        assert_eq!(shown[4], "not a cycle: d(z) = Sq(3) x_0,0");

        // Formatting and parsing round-trip.
        let (s, t, y) = (cycles[2].0 + 1, cycles[2].1, lifts[2].as_ref().unwrap());
        let text = element_string(&res, s, t, y);
        assert_eq!(element(&text).2, *y);
    }

    #[test]
    fn mixed_bidegrees_are_rejected() {
        let res = ext::utils::construct(("S_2", "milnor"), None).unwrap();
        assert!(parse(&res, "Sq(1) x_0,0 + Sq(2) x_0,0").is_err());
        assert!(parse(&res, "Sq(1) x_0,0 + x_1,1").is_err());
        assert!(parse(&res, "Q_0 x_0,0").is_err());
    }
}
