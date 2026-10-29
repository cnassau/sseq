//! Products in Ext and the chain maps that realise them, read off a resolution.

use std::fmt::Write;

use algebra::{Algebra, module::Module};
use ext::{
    chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex},
    ext_algebra::ExtAlgebra,
};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeElement};

use crate::{Res, elements::element_string};

/// What to multiply by.
pub enum Multiplier {
    /// A named filtration-one class, such as `h_1`, given by its dual algebra element.
    Named {
        name: String,
        op_deg: i32,
        op_idx: usize,
    },
    /// A class of `Ext(k, k)`; only for resolutions of `k` itself.
    Class(BidegreeElement),
}

impl Multiplier {
    /// The bidegree of the multiplier.
    pub fn degree(&self) -> Bidegree {
        match self {
            Self::Named { op_deg, .. } => Bidegree::s_t(1, *op_deg),
            Self::Class(x) => x.degree(),
        }
    }

    /// A name for the output header.
    fn describe(&self) -> String {
        match self {
            Self::Named { name, .. } => name.clone(),
            Self::Class(x) => format!("the class {x}"),
        }
    }

    /// The multiplier as a class of `Ext(k, k)`, which needs `res` to resolve `k` and to be computed
    /// through the multiplier's bidegree.
    pub fn to_class(&self, res: &Res) -> anyhow::Result<BidegreeElement> {
        match self {
            Self::Class(x) => Ok(x.clone()),
            Self::Named {
                name,
                op_deg,
                op_idx,
            } => {
                anyhow::ensure!(
                    is_unit(res),
                    "{name} is a class of Ext only over the ground field, and this module is not \
                     the ground field"
                );
                // The product with the unit class is the class itself.
                let unit = res.try_filtration_one_product(*op_deg, *op_idx, Bidegree::s_t(0, 0))?;
                Ok(BidegreeElement::new(
                    Bidegree::s_t(1, *op_deg),
                    FpVector::from_slice(res.prime(), &unit[0]),
                ))
            }
        }
    }
}

/// The named filtration-one products of the algebra `res` is over, as `(name, op_deg, op_idx)`.
pub fn named(res: &Res) -> Vec<(String, i32, usize)> {
    res.algebra().default_filtration_one_products()
}

/// Whether `res` resolves the ground field `k`, so that products by any class of `Ext(k, k)` are
/// available.
pub fn is_unit(res: &Res) -> bool {
    res.target().max_s() == 1 && res.target().module(0).is_unit()
}

/// Parse `by` as a filtration-one name (`h_1` or `h1`) or as `n s i`, the `i`-th generator in stem
/// `n` and filtration `s`.
pub fn parse(res: &Res, by: &str, coords: Option<&[u32]>) -> anyhow::Result<Multiplier> {
    let names = named(res);
    let normalized = by.trim().replace('_', "");
    if let Some((name, op_deg, op_idx)) = names
        .iter()
        .find(|(name, ..)| name.replace('_', "") == normalized)
    {
        anyhow::ensure!(
            coords.is_none(),
            "coords only apply to a bidegree, not to {name}"
        );
        return Ok(Multiplier::Named {
            name: name.clone(),
            op_deg: *op_deg,
            op_idx: *op_idx,
        });
    }

    let fields: Vec<i32> = by
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| {
            let names: Vec<&str> = names.iter().map(|(name, ..)| name.as_str()).collect();
            anyhow::anyhow!(
                "`by` must be one of {names:?}, or a bidegree \"n s\" (with coords) or \"n s i\""
            )
        })?;
    let (b, idx) = match fields[..] {
        [n, s] => (Bidegree::n_s(n, s), None),
        [n, s, i] if i >= 0 => (Bidegree::n_s(n, s), Some(i as usize)),
        _ => anyhow::bail!("a bidegree is \"n s\" or \"n s i\""),
    };
    anyhow::ensure!(
        is_unit(res),
        "products by arbitrary classes need a resolution of the ground field (such as S_2 or \
         tmf2); this module only has the named products"
    );
    anyhow::ensure!(
        res.has_computed_bidegree(b),
        "bidegree (n, s) = ({}, {}) has not been resolved yet",
        b.n(),
        b.s()
    );
    let dim = res.number_of_gens_in_bidegree(b);
    let p = res.prime();
    let vec = match (idx, coords) {
        (Some(i), None) => {
            anyhow::ensure!(
                i < dim,
                "Ext there has dimension {dim}, so no generator {i}"
            );
            let mut v = FpVector::new(p, dim);
            v.set_entry(i, 1);
            v
        }
        (None, Some(coords)) => {
            anyhow::ensure!(
                coords.len() == dim,
                "Ext there has dimension {dim}, but {} coords were given",
                coords.len()
            );
            FpVector::from_slice(p, coords)
        }
        _ => anyhow::bail!("give either \"n s i\" or \"n s\" together with coords"),
    };
    anyhow::ensure!(!vec.is_zero(), "the multiplier is zero");
    Ok(Multiplier::Class(BidegreeElement::new(b, vec)))
}

/// The nonzero products `x · g` for every generator `g` with stem at most `max_n` and filtration at
/// most `max_s`, as `n s i -> n' s' : coordinates` lines.
///
/// `res` must be computed through the stem of `(max_n, max_s)` shifted by the degree of `x`.
/// `ext` is the ring the products of a [`Multiplier::Class`] come from.
pub fn table(
    res: &Res,
    ext: &ExtAlgebra<Res>,
    x: &Multiplier,
    max_n: i32,
    max_s: i32,
) -> anyhow::Result<String> {
    let shift = x.degree();
    let mut out = format!(
        "Products by {} in bidegree (n, s) = ({}, {}). Each line `n s i -> n' s' : c_0 c_1 ...` \
         gives the product with generator i of (n, s) in the generator basis of (n', s'). \
         Products that vanish are left out.\n",
        x.describe(),
        shift.n(),
        shift.s()
    );
    for b in res.iter_stem() {
        if b.n() > max_n || b.s() > max_s || res.number_of_gens_in_bidegree(b) == 0 {
            continue;
        }
        let target = b + shift;
        let rows: Vec<Vec<u32>> = match x {
            Multiplier::Named { op_deg, op_idx, .. } => {
                res.try_filtration_one_product(*op_deg, *op_idx, b)?
            }
            Multiplier::Class(x) => {
                let matrix = ext.multiply_into(x, b).ok_or_else(|| {
                    anyhow::anyhow!("the product into {target} is out of the computed range")
                })?;
                (0..matrix.rows())
                    .map(|i| matrix.row(i).iter().collect())
                    .collect()
            }
        };
        for (i, row) in rows.iter().enumerate() {
            if row.iter().any(|&c| c != 0) {
                let coords: Vec<String> = row.iter().map(u32::to_string).collect();
                writeln!(
                    out,
                    "{} {} {i} -> {} {} : {}",
                    b.n(),
                    b.s(),
                    target.n(),
                    target.s(),
                    coords.join(" ")
                )
                .unwrap();
            }
        }
    }
    Ok(out)
}

/// The chain map `gamma` lifting `x` on every generator with stem at most `max_n` and filtration
/// at most `max_s`, as `n s generator : value` lines.
///
/// `res` must be computed through `(max_n, max_s)`.
pub fn chain_map_table(
    res: &Res,
    ext: &ExtAlgebra<Res>,
    x: &BidegreeElement,
    max_n: i32,
    max_s: i32,
) -> String {
    let shift = x.degree();
    let gamma = ext.class_product_map(x, Bidegree::n_s(max_n, max_s));
    let mut out = format!(
        "The chain map gamma: C_s -> C_(s - {s}) lifting the class {x} in (n, s) = ({n}, {s}); it \
         lowers the internal degree t by {t}. Each line `n s generator : gamma(generator)` gives \
         gamma on a generator x_s,t of C_s, with the index [i] when there are several, in the \
         Milnor basis. Generators with s < {s} are outside its domain and left out. gamma is the \
         lift the library computes; other lifts differ from it by a chain homotopy.\n",
        n = shift.n(),
        s = shift.s(),
        t = shift.t(),
    );
    let mut bidegrees: Vec<Bidegree> = res
        .iter_stem()
        .filter(|b| b.n() <= max_n && shift.s() <= b.s() && b.s() <= max_s)
        .collect();
    bidegrees.sort_by_key(|b| (b.n(), b.s()));
    for b in bidegrees {
        for i in 0..res.number_of_gens_in_bidegree(b) {
            let target = b - shift;
            // Below the target's bottom degree there is nothing to map to, and no stored value.
            let value = if target.t() < res.module(target.s()).min_degree() {
                "0".to_string()
            } else {
                element_string(
                    res,
                    target.s(),
                    target.t(),
                    gamma.get_map(b.s()).output(b.t(), i),
                )
            };
            writeln!(
                out,
                "{} {} {} : {value}",
                b.n(),
                b.s(),
                crate::elements::generator_name(res, b.s(), b.t(), i)
            )
            .unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// The resolution of `module` through stem `n` and filtration `s`, with its ring.
    fn resolve(module: &str, n: i32, s: i32) -> (Arc<Res>, ExtAlgebra<Res>) {
        let res = Arc::new(ext::utils::construct((module, "milnor"), None).unwrap());
        res.compute_through_stem(Bidegree::n_s(n, s));
        let ext = ExtAlgebra::new(Arc::clone(&res));
        (res, ext)
    }

    /// The product lines of `table`, without the header.
    fn lines(res: &Res, ext: &ExtAlgebra<Res>, by: &str, n: i32, s: i32) -> Vec<String> {
        let x = parse(res, by, None).unwrap();
        let table = table(res, ext, &x, n, s).unwrap();
        table.lines().skip(1).map(str::to_string).collect()
    }

    #[test]
    fn sphere_at_two() {
        let (res, ext) = resolve("S_2", 12, 6);
        let h0 = lines(&res, &ext, "h0", 6, 4);
        let h1 = lines(&res, &ext, "h_1", 6, 4);
        assert!(h0.contains(&"0 1 0 -> 0 2 : 1".to_string()));
        assert!(h0.contains(&"3 1 0 -> 3 2 : 1".to_string()));
        assert!(h1.contains(&"1 1 0 -> 2 2 : 1".to_string()));
        // h0 h1 = 0 and h1 h2 = 0.
        assert!(!h0.iter().any(|l| l.starts_with("1 1 0 ")));
        assert!(!h1.iter().any(|l| l.starts_with("3 1 0 ")));
    }

    /// The products by a named class agree with those by the same class given by its bidegree.
    #[rstest::rstest]
    #[case("S_2", "h_2", "3 1 0")]
    #[case("tmf2", "h_1", "1 1 0")]
    #[case("S_3", "h_0", "3 1 0")]
    #[case("S_3", "a_0", "0 1 0")]
    fn named_matches_class(#[case] module: &str, #[case] name: &str, #[case] class: &str) {
        let (res, ext) = resolve(module, 30, 8);
        let p = res.prime();
        // Up to sign: compare lines with every coordinate replaced by whether it is zero.
        let normalize = |lines: Vec<String>| -> Vec<String> {
            lines
                .into_iter()
                .map(|l| {
                    let (head, coords) = l.split_once(" : ").unwrap();
                    let coords: Vec<&str> = coords
                        .split(' ')
                        .map(|c| if c == "0" { "0" } else { "*" })
                        .collect();
                    format!("{head} : {}", coords.join(" "))
                })
                .collect()
        };
        let by_name = normalize(lines(&res, &ext, name, 20, 6));
        let by_class = normalize(lines(&res, &ext, class, 20, 6));
        assert!(!by_name.is_empty());
        assert_eq!(by_name, by_class, "at p = {p}");
    }

    #[test]
    fn class_products_need_the_unit() {
        let (res, _) = resolve("C2", 4, 2);
        assert!(!is_unit(&res));
        assert!(parse(&res, "h0", None).is_ok());
        assert!(parse(&res, "0 1 0", None).is_err());
    }

    #[test]
    fn parsing() {
        let (res, _) = resolve("S_2", 4, 2);
        assert!(parse(&res, "h_9", None).is_err());
        assert!(parse(&res, "0 1 1", None).is_err());
        assert!(parse(&res, "0 1", Some(&[1])).is_ok());
        assert!(parse(&res, "0 1", Some(&[0])).is_err());
        assert!(parse(&res, "0 1", None).is_err());
    }

    #[test]
    fn chain_maps() {
        let (res, ext) = resolve("S_2", 16, 8);
        let h0 = parse(&res, "h_0", None).unwrap().to_class(&res).unwrap();
        let table = chain_map_table(&res, &ext, &h0, 16, 8);
        // gamma lifts h0, so it sends the generator of C_1 in degree 1 to the unit generator.
        assert!(table.contains("\n0 1 x_1,1 : 1 x_0,0\n"), "{table}");
        // On h4 it covers multiplication h0 h4: a chain map composed with d is d composed with it,
        // so its value on x_2,17 has a term hitting x_1,16.
        let line = table
            .lines()
            .find(|l| l.starts_with("15 2 x_2,17 : "))
            .unwrap();
        assert!(line.contains("1 x_1,16"), "{line}");
        // Generators below the class's filtration are outside the domain.
        assert!(!table.lines().any(|l| l.starts_with("0 0 ")));

        let x = parse(&res, "0 1 0", None).unwrap().to_class(&res).unwrap();
        assert_eq!(chain_map_table(&res, &ext, &x, 16, 8), table);
    }

    /// A class of positive stem maps generators of smaller stem to zero, for degree reasons.
    #[test]
    fn chain_map_below_its_degree() {
        let (res, ext) = resolve("S_2", 16, 8);
        let h0_5_h4 = parse(&res, "15 6 0", None).unwrap().to_class(&res).unwrap();
        let table = chain_map_table(&res, &ext, &h0_5_h4, 16, 8);
        assert!(table.contains("\n0 6 x_6,6 : 0\n"), "{table}");
        // On h0^5 h4 itself it lands on the unit generator.
        assert!(table.contains("\n15 6 x_6,21 : 1 x_0,0\n"), "{table}");
    }
}
