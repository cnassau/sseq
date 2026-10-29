//! Rendering resolution data as Sage source for `sseq.sage`.

use std::fmt::Write;

use algebra::{
    Algebra, MilnorAlgebra, SteenrodKind,
    milnor_algebra::{MilnorBasisElement, MilnorProfile, PPartEntry},
    module::{FreeModule, Module},
};
use ext::chain_complex::{ChainComplex, FreeChainComplex};
use sseq::coordinates::Bidegree;

use crate::Res;

/// Sage expression for the algebra `res` is over, e.g.
/// `SteenrodAlgebra(2, basis='milnor', profile=(2, 1), truncation_type=0)`.
pub fn sage_algebra(algebra: &MilnorAlgebra) -> anyhow::Result<String> {
    if algebra.kind() != SteenrodKind::Classical {
        anyhow::bail!("only sub-Hopf-algebras of the classical Steenrod algebra are supported");
    }
    let p = algebra.prime();
    let profile = algebra.profile();
    if profile.is_trivial() {
        return Ok(format!("SteenrodAlgebra({p}, basis='milnor')"));
    }
    let e = tuple(profile.p_part.iter().map(|&x| exponent(x)));
    let truncation = if profile.truncated { "0" } else { "Infinity" };
    let profile = if p == 2 {
        e
    } else {
        format!("({e}, {})", q_profile(profile))
    };
    Ok(format!(
        "SteenrodAlgebra({p}, basis='milnor', profile={profile}, truncation_type={truncation})"
    ))
}

/// A profile exponent; ours marks an unbounded one by `PPartEntry::MAX`.
fn exponent(x: PPartEntry) -> String {
    if x == PPartEntry::MAX {
        "Infinity".to_string()
    } else {
        x.to_string()
    }
}

/// Sage's `k` profile, which marks `Q_i` present by 2 and absent by 1.
///
/// Entries past the end of the tuple follow `truncation_type`, so the tuple runs up to the last bit
/// of our mask that disagrees with it, and at least as far as the p-part.
fn q_profile(profile: &MilnorProfile) -> String {
    let present = |i: u32| profile.q_part & (1 << i) != 0;
    let len = (0..u32::BITS)
        .filter(|&i| present(i) == profile.truncated)
        .map(|i| i + 1)
        .max()
        .unwrap_or(0)
        .max(profile.p_part.len() as u32 + 1);
    tuple((0..len).map(|i| if present(i) { "2" } else { "1" }))
}

/// A Python tuple literal, with the trailing comma a 1-tuple needs.
fn tuple<T: std::fmt::Display>(items: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = items.into_iter().map(|x| x.to_string()).collect();
    match items.len() {
        1 => format!("({},)", items[0]),
        _ => format!("({})", items.join(", ")),
    }
}

/// The key Sage's Milnor basis uses for `elt`: the P-part at p = 2, `(Q-part, P-part)` otherwise.
pub fn milnor_key(elt: &MilnorBasisElement, generic: bool) -> String {
    let mut p_part: Vec<PPartEntry> = elt.p_part.iter().collect();
    while p_part.last() == Some(&0) {
        p_part.pop();
    }
    let p_part = tuple(p_part);
    if generic {
        let q_part = (0..u32::BITS).filter(|&i| elt.q_part & (1 << i) != 0);
        format!("({}, {p_part})", tuple(q_part))
    } else {
        p_part
    }
}

/// Sage code passing the generators and differentials of `res` in the given range to the
/// `SseqResolution` named `var`.
///
/// The range must already be computed.
pub fn sage_data(res: &Res, var: &str, s_min: i32, s_max: i32, t_max: i32) -> String {
    let algebra = res.algebra();
    let algebra: &MilnorAlgebra = (&*algebra)
        .try_into()
        .expect("resolutions use the Milnor basis");
    let generic = algebra.prime() != 2;

    let mut ngens = String::new();
    let mut diffs = String::new();
    for s in s_min..=s_max {
        let source = res.module(s);
        for t in source.min_degree()..=t_max {
            let n = res.number_of_gens_in_bidegree(Bidegree::s_t(s, t));
            if n == 0 {
                continue;
            }
            write!(ngens, "({s}, {t}): {n}, ").unwrap();
            let d = res.differential(s);
            for i in 0..n {
                write!(diffs, "({s}, {t}, {i}): [").unwrap();
                if s > 0 {
                    let target = res.module(s - 1);
                    write_terms(&mut diffs, algebra, &target, t, d.output(t, i), generic);
                }
                diffs.push_str("], ");
            }
        }
    }
    format!("{var}.add_data(({s_min}, {s_max}), {t_max}, {{{ngens}}}, {{{diffs}}})")
}

/// Write the terms of `v`, an element of `target` in degree `t`, as `(c, key, (t', j))` tuples
/// (`c` only when `generic`).
fn write_terms(
    out: &mut String,
    algebra: &MilnorAlgebra,
    target: &FreeModule<algebra::SteenrodAlgebra>,
    t: i32,
    v: &fp::vector::FpVector,
    generic: bool,
) {
    for (index, c) in v.iter_nonzero() {
        let opgen = target.index_to_op_gen(t, index);
        let elt = algebra.basis_element_from_index(opgen.operation_degree, opgen.operation_index);
        let key = milnor_key(&elt, generic);
        let g = (opgen.generator_degree, opgen.generator_index);
        if generic {
            write!(out, "({c}, {key}, {g:?}), ").unwrap();
        } else {
            write!(out, "({key}, {g:?}), ").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuples() {
        assert_eq!(tuple(Vec::<u32>::new()), "()");
        assert_eq!(tuple([3]), "(3,)");
        assert_eq!(tuple([2, 1]), "(2, 1)");
    }

    #[test]
    fn q_profiles() {
        let profile = |truncated, q_part, p_part: Vec<PPartEntry>| MilnorProfile {
            truncated,
            q_part,
            p_part,
        };
        assert_eq!(q_profile(&profile(true, 0b11, vec![1])), "(2, 2)");
        assert_eq!(q_profile(&profile(true, 0b1, vec![])), "(2,)");
        assert_eq!(q_profile(&profile(false, !0, vec![1, 1])), "(2, 2, 2)");
        assert_eq!(q_profile(&profile(false, !0b10, vec![])), "(2, 1)");
    }

    /// The data for the module `module` through `(s_max, t_max)`.
    fn data(module: &str, s_max: i32, t_max: i32) -> String {
        let res = ext::utils::construct((module, "milnor"), None).unwrap();
        res.compute_through_bidegree(Bidegree::s_t(s_max, t_max));
        sage_data(&res, "R", 0, s_max, t_max)
    }

    #[test]
    fn sphere_at_two() {
        let code = data("S_2", 2, 4);
        // h0, h1, h2, and h0^2 with d = Sq(1) h0.
        for term in [
            "(1, 1, 0): [((1,), (0, 0)), ]",
            "(1, 2, 0): [((2,), (0, 0)), ]",
            "(1, 4, 0): [((4,), (0, 0)), ]",
            "(2, 2, 0): [((1,), (1, 0)), ]",
        ] {
            assert!(code.contains(term), "{term} missing from {code}");
        }
        assert!(code.starts_with("R.add_data((0, 2), 4, {(0, 0): 1, (1, 1): 1, (1, 2): 1, "));
    }

    #[test]
    fn sphere_at_three() {
        let code = data("S_3", 1, 4);
        // a0 with d = Q_0, and h0 with d = P(1).
        assert!(
            code.contains("(1, 1, 0): [(1, ((0,), ()), (0, 0)), ]"),
            "{code}"
        );
        assert!(
            code.contains("(1, 4, 0): [(1, ((), (1,)), (0, 0)), ]"),
            "{code}"
        );
    }

    #[test]
    fn algebras() {
        let algebra = |module| {
            let res = ext::utils::construct((module, "milnor"), None).unwrap();
            let algebra = res.algebra();
            sage_algebra((&*algebra).try_into().unwrap()).unwrap()
        };
        assert_eq!(algebra("S_2"), "SteenrodAlgebra(2, basis='milnor')");
        assert_eq!(
            algebra("ko"),
            "SteenrodAlgebra(2, basis='milnor', profile=(2, 1), truncation_type=0)"
        );
    }
}
