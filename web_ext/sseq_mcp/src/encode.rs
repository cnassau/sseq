//! Rendering resolution data as Sage source for `sseq.sage`.

use std::fmt::Write;

use algebra::{
    Algebra, MilnorAlgebra, SteenrodKind,
    milnor_algebra::{MilnorBasisElement, MilnorProfile, PPartEntry},
    module::{FreeModule, Module},
};
use ext::chain_complex::{
    AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex,
};
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

/// Sage code assigning to `var` the module `res` resolves, as a `SseqSteenrodModule` holding the
/// action of every Milnor basis element on every basis element through degree `t_max`.
///
/// `t_max` defaults to the top degree of a finite module and is capped by it; an infinite module
/// needs one.
pub fn sage_steenrod_module(
    res: &Res,
    name: &str,
    var: &str,
    t_max: Option<i32>,
) -> anyhow::Result<String> {
    let target = res.target();
    anyhow::ensure!(
        target.max_s() == 1,
        "{name} is the cofiber of a map, a two-term chain complex rather than a single module"
    );
    let module = target.module(0);
    let algebra = res.algebra();
    let milnor: &MilnorAlgebra = (&*algebra).try_into()?;
    let generic = milnor.prime() != 2;
    let min = module.min_degree();
    let top = module.max_degree();
    let t_max = match (t_max, top) {
        (Some(t), Some(top)) => t.min(top),
        (Some(t), None) => t,
        (None, Some(top)) => top,
        (None, None) => anyhow::bail!("{name} is infinite, so give t_max"),
    };
    anyhow::ensure!(
        t_max >= min && t_max - min <= MAX_MODULE_RANGE,
        "t_max must lie between the bottom degree {min} and {}",
        min + MAX_MODULE_RANGE
    );
    let complete = top.is_some_and(|top| t_max >= top);
    module.compute_basis(t_max);
    algebra.compute_basis(t_max - min);

    let mut names = String::new();
    let mut actions = String::new();
    for t in min..=t_max {
        for i in 0..module.dimension(t) {
            let basis_name = python_str(&module.basis_element_to_string(t, i));
            write!(names, "({t}, {i}): {basis_name}, ").unwrap();
            for d in 1..=t_max - t {
                for op in 0..algebra.dimension(d) {
                    let mut result =
                        fp::vector::FpVector::new(milnor.prime(), module.dimension(t + d));
                    module.act_on_basis(result.as_slice_mut(), 1, d, op, t, i);
                    if result.is_zero() {
                        continue;
                    }
                    let key = milnor_key(&milnor.basis_element_from_index(d, op), generic);
                    write!(actions, "({key}, ({t}, {i})): [").unwrap();
                    for (j, c) in result.iter_nonzero() {
                        write!(actions, "({c}, ({}, {j})), ", t + d).unwrap();
                    }
                    actions.push_str("], ");
                }
            }
        }
    }
    let complete = if complete { "True" } else { "False" };
    Ok(format!(
        "{var} = sseq_steenrod_module({name}, {p}, {algebra}, {{{names}}}, {t_max}, {complete}, \
         {{{actions}}})",
        name = python_str(name),
        p = milnor.prime(),
        algebra = sage_algebra(milnor)?,
    ))
}

/// Largest range of degrees a `SseqSteenrodModule` covers. The action table grows with the number
/// of Milnor basis elements across the range, so this keeps replies to a size worth relaying.
const MAX_MODULE_RANGE: i32 = 64;

/// A Python string literal for `s`.
fn python_str(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            c if c.is_ascii_graphic() || c == ' ' => out.push(c),
            c if (c as u32) < 0x10000 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => write!(out, "\\U{:08x}", c as u32).unwrap(),
        }
    }
    out.push('\'');
    out
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

    #[test]
    fn python_strings() {
        assert_eq!(python_str("x0"), "'x0'");
        assert_eq!(python_str("a'b\\"), "'a\\'b\\\\'");
        assert_eq!(python_str("\u{3b1}\n"), "'\\u03b1\\u000a'");
    }

    /// The module of a resolution, by name.
    fn module(name: &str, t_max: Option<i32>) -> anyhow::Result<String> {
        let res = ext::utils::construct((name, "milnor"), None).unwrap();
        sage_steenrod_module(&res, name, "M", t_max)
    }

    #[test]
    fn steenrod_modules() {
        // Ceta: cells in degrees 0 and 2 joined by Sq(2).
        let ceta = module("Ceta", None).unwrap();
        assert!(
            ceta.contains(", 2, True, {((2,), (0, 0)): [(1, (2, 0)), ], })"),
            "{ceta}"
        );
        // Calpha at p = 3: P(1) joins the cells in degrees 0 and 4.
        let calpha = module("Calpha", None).unwrap();
        assert!(
            calpha.contains("(((), (1,)), (0, 0)): [(1, (4, 0)), ]"),
            "{calpha}"
        );
        // Infinite modules need t_max and are marked incomplete.
        assert!(module("RP_inf", None).is_err());
        assert!(module("RP_inf", Some(8)).unwrap().contains(", 8, False, {"));
        // A cofiber is not a single module.
        assert!(module("C2v14", None).is_err());
    }

    /// Every built-in module either builds or is refused with an error, never a panic.
    #[test]
    fn every_module() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../ext/steenrod_modules");
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            // Building the p = 7 algebra overflows an i32 in MilnorAlgebra's exterior basis
            // generation, which panics in debug builds.
            if name == "S_7" {
                continue;
            }
            let result = module(&name, Some(16));
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let cofiber = !json["cofiber"].is_null();
            assert_eq!(result.is_err(), cofiber, "{name}: {result:?}");
        }
    }
}
