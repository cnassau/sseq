use std::sync::Arc;

use algebra::{
    MilnorAlgebra, SteenrodAlgebra, SteenrodKind,
    milnor_algebra::MilnorProfile,
    module::{FDModule, steenrod_module},
};
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    resolution::Resolution,
    utils::{construct_groebner, construct_standard},
};
use rstest::rstest;
use sseq::coordinates::Bidegree;

#[rstest]
#[trace]
#[case("S_2", 30, 15)]
#[case("C2", 30, 15)]
#[case("Joker", 30, 15)]
#[case("Csigma", 30, 15)]
#[case("Z", 30, 10)]
#[case("S_3", 60, 10)]
#[case("C3v1", 60, 10)]
#[case("S_5", 120, 6)]
#[case("C5v1", 120, 6)]
fn compare_stem(#[case] module_name: &str, #[case] n: i32, #[case] s: i32) {
    let max = Bidegree::n_s(n, s);
    let a = construct_standard::<false, _, _>(module_name, None).unwrap();
    let b = construct_groebner(module_name, None).unwrap();

    a.compute_through_stem(max);
    b.compute_through_stem(max);

    assert_eq!(a.graded_dimension_string(), b.graded_dimension_string());
}

#[rstest]
#[trace]
#[case("S_2", 20)]
#[case("Joker", 20)]
#[case("S_3", 40)]
fn compare_bidegree(#[case] module_name: &str, #[case] t: i32) {
    let max = Bidegree::s_t(t, t);
    let a = construct_standard::<false, _, _>(module_name, None).unwrap();
    let b = construct_groebner(module_name, None).unwrap();

    a.compute_through_bidegree(max);
    b.compute_through_bidegree(max);

    assert_eq!(a.graded_dimension_string(), b.graded_dimension_string());
}

/// The other kinds of Steenrod algebra have no Adem basis, so compare against the standard
/// resolution over the same Milnor algebra.
#[rstest]
#[trace]
#[case(2, SteenrodKind::MotivicModTau, 20, 8)]
#[case(3, SteenrodKind::ReducedPowers, 60, 6)]
fn compare_kind(#[case] p: u32, #[case] kind: SteenrodKind, #[case] n: i32, #[case] s: i32) {
    let max = Bidegree::n_s(n, s);
    let name = format!("S_{p}");
    let milnor = match kind {
        SteenrodKind::MotivicModTau => {
            MilnorAlgebra::motivic_mod_tau(MilnorProfile::default(), false)
        }
        _ => MilnorAlgebra::reduced_powers(
            fp::prime::ValidPrime::new(p),
            MilnorProfile::default(),
            false,
        ),
    };
    let algebra = Arc::new(SteenrodAlgebra::MilnorAlgebra(milnor));
    let module = steenrod_module::erase(FDModule::new(
        algebra,
        name.clone(),
        bivec::BiVec::from_vec(0, vec![1]),
    ));
    let cc: ext::CCC = FiniteChainComplex::ccdz(Arc::new(module));
    let a = Resolution::new(Arc::new(cc));
    let b = construct_groebner(name.as_str(), Some(kind)).unwrap();

    a.compute_through_stem(max);
    b.compute_through_stem(max);

    assert_eq!(a.graded_dimension_string(), b.graded_dimension_string());
}

fn rank(matrix: &[Vec<u32>], p: u32) -> usize {
    let mut rows: Vec<Vec<u32>> = matrix.to_vec();
    let mut rank = 0;
    let cols = rows.first().map_or(0, Vec::len);
    for col in 0..cols {
        let Some(pivot) = (rank..rows.len()).find(|&r| !rows[r][col].is_multiple_of(p)) else {
            continue;
        };
        rows.swap(rank, pivot);
        let inv = (1..p).find(|x| x * rows[rank][col] % p == 1).unwrap();
        let pivot_row: Vec<u32> = rows[rank].iter().map(|x| x * inv % p).collect();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && row[col] % p != 0 {
                let c = row[col];
                for (x, y) in row.iter_mut().zip(&pivot_row) {
                    *x = (*x + (p - c) * y) % p;
                }
            }
        }
        rank += 1;
    }
    rank
}

/// The products by the elements of filtration one are read off the differentials, so this checks
/// them beyond the dimensions. Bases may differ, so compare ranks.
#[rstest]
#[trace]
#[case("S_2", 30, 10, &[(1, 0), (2, 0), (4, 0)])]
#[case("Joker", 25, 8, &[(1, 0), (2, 0)])]
#[case("S_3", 60, 8, &[(1, 0), (4, 0)])]
fn compare_products(
    #[case] module_name: &str,
    #[case] n: i32,
    #[case] s: i32,
    #[case] ops: &[(i32, usize)],
) {
    let max = Bidegree::n_s(n, s);
    let a = construct_standard::<false, _, _>(module_name, None).unwrap();
    let b = construct_groebner(module_name, None).unwrap();
    a.compute_through_stem(max);
    b.compute_through_stem(max);
    let p = fp::prime::Prime::as_u32(a.prime());

    let mut checked = 0;
    for &(op_deg, op_idx) in ops {
        for source in a.iter_stem() {
            let target = source + Bidegree::s_t(1, op_deg);
            if !a.has_computed_bidegree(target) || !b.has_computed_bidegree(target) {
                continue;
            }
            let pa = a.filtration_one_product(op_deg, op_idx, source);
            let pb = b.filtration_one_product(op_deg, op_idx, source);
            match (pa, pb) {
                (Some(pa), Some(pb)) => {
                    assert_eq!(rank(&pa, p), rank(&pb, p), "{op_deg} on {source}");
                    checked += 1;
                }
                (pa, pb) => assert_eq!(pa.is_some(), pb.is_some(), "{op_deg} on {source}"),
            }
        }
    }
    assert!(checked > 0);
}

/// A module whose bottom cell is not in degree zero.
#[test]
fn shifted_module() {
    let json = serde_json::json!({
        "p": 2,
        "type": "finite dimensional module",
        "gens": {"x3": 3, "x4": 4, "x6": 6},
        "actions": ["Sq1 x3 = x4", "Sq2 x4 = x6"],
    });
    let max = Bidegree::s_t(8, 30);
    let a = construct_standard::<false, _, _>((json.clone(), "milnor"), None).unwrap();
    let b = construct_groebner((json, "milnor"), None).unwrap();
    a.compute_through_bidegree(max);
    b.compute_through_bidegree(max);
    assert_eq!(a.graded_dimension_string(), b.graded_dimension_string());
}
