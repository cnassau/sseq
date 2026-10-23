use algebra::{
    Algebra, MilnorAlgebra, SteenrodKind,
    milnor_algebra::{MilnorBasisElement, MilnorProfile, PPart},
};
use fp::{prime::ValidPrime, vector::FpVector};
use rstest::rstest;

use crate::{Mono, Params};

fn milnor_algebra(p: u32, kind: SteenrodKind) -> MilnorAlgebra {
    let vp = ValidPrime::new(p);
    match kind {
        SteenrodKind::Classical => MilnorAlgebra::new(vp, false),
        SteenrodKind::MotivicModTau => {
            MilnorAlgebra::motivic_mod_tau(MilnorProfile::default(), false)
        }
        SteenrodKind::ReducedPowers => {
            MilnorAlgebra::reduced_powers(vp, MilnorProfile::default(), false)
        }
    }
}

fn to_mono(params: &Params, elt: &MilnorBasisElement) -> Mono {
    let r: Vec<u32> = elt.p_part.iter().collect();
    params.encode(elt.q_part, &r).unwrap()
}

fn to_index(algebra: &MilnorAlgebra, params: &Params, m: Mono) -> usize {
    let (q_part, r) = params.decode(m);
    algebra.basis_element_to_index(&MilnorBasisElement {
        q_part,
        p_part: PPart::from_slice(&r),
        degree: params.degree(m),
    })
}

fn basis(algebra: &MilnorAlgebra, params: &Params, degree: i32) -> Vec<Mono> {
    algebra.compute_basis(degree);
    (0..algebra.dimension(degree))
        .map(|i| to_mono(params, &algebra.basis_element_from_index(degree, i)))
        .collect()
}

#[rstest]
#[case(2, SteenrodKind::Classical, 40)]
#[case(2, SteenrodKind::MotivicModTau, 40)]
#[case(3, SteenrodKind::Classical, 70)]
#[case(3, SteenrodKind::ReducedPowers, 70)]
#[case(5, SteenrodKind::Classical, 120)]
fn encoding_roundtrip(#[case] p: u32, #[case] kind: SteenrodKind, #[case] max: i32) {
    let algebra = milnor_algebra(p, kind);
    let params = Params::new(p, kind);
    for degree in 0..=max {
        for (i, m) in basis(&algebra, &params, degree).into_iter().enumerate() {
            assert_eq!(params.degree(m), degree);
            assert_eq!(to_index(&algebra, &params, m), i);
        }
    }
}

#[rstest]
#[case(2, SteenrodKind::Classical, 34)]
#[case(2, SteenrodKind::MotivicModTau, 34)]
#[case(3, SteenrodKind::Classical, 60)]
#[case(3, SteenrodKind::ReducedPowers, 60)]
#[case(5, SteenrodKind::Classical, 90)]
fn product_matches_milnor_algebra(#[case] p: u32, #[case] kind: SteenrodKind, #[case] max: i32) {
    let algebra = milnor_algebra(p, kind);
    let params = Params::new(p, kind);
    algebra.compute_basis(max);
    for d1 in 0..=max {
        for d2 in 0..=max - d1 {
            let dim = algebra.dimension(d1 + d2);
            for a in basis(&algebra, &params, d1) {
                for b in basis(&algebra, &params, d2) {
                    let mut expected = FpVector::new(ValidPrime::new(p), dim);
                    let ea = algebra.basis_element_from_index(d1, to_index(&algebra, &params, a));
                    let eb = algebra.basis_element_from_index(d2, to_index(&algebra, &params, b));
                    algebra.multiply(expected.as_slice_mut(), 1, ea, eb);

                    let mut actual = FpVector::new(ValidPrime::new(p), dim);
                    params.mul(a, b, |m, c| {
                        assert_eq!(params.degree(m), d1 + d2);
                        actual.add_basis_element(to_index(&algebra, &params, m), c)
                    });
                    assert_eq!(
                        actual,
                        expected,
                        "{} * {}",
                        params.format(a),
                        params.format(b)
                    );
                }
            }
        }
    }
}

/// The leading term of a product is the product of leading terms in the associated graded
/// algebra, and if that vanishes, every term has larger weight.
#[rstest]
#[case(2, SteenrodKind::Classical, 40)]
#[case(2, SteenrodKind::MotivicModTau, 40)]
#[case(3, SteenrodKind::Classical, 70)]
#[case(3, SteenrodKind::ReducedPowers, 70)]
#[case(5, SteenrodKind::Classical, 100)]
fn leading_terms_multiply(#[case] p: u32, #[case] kind: SteenrodKind, #[case] max: i32) {
    let algebra = milnor_algebra(p, kind);
    let params = Params::new(p, kind);
    for d1 in 0..=max {
        for d2 in 0..=max - d1 {
            for a in basis(&algebra, &params, d1) {
                for b in basis(&algebra, &params, d2) {
                    let mut terms: Vec<(Mono, u32)> = Vec::new();
                    params.mul(a, b, |m, c| terms.push((m, c)));
                    terms.sort();
                    let mut combined: Vec<(Mono, u32)> = Vec::new();
                    for (m, c) in terms {
                        match combined.last_mut() {
                            Some((l, d)) if *l == m => *d = (*d + c) % p,
                            _ => combined.push((m, c)),
                        }
                    }
                    combined.retain(|&(_, c)| c != 0);
                    let ctx = format!("{} * {}", params.format(a), params.format(b));
                    match params.lf_mul(a, b) {
                        Some(m) => {
                            assert_eq!(combined[0].0, m, "{ctx}");
                            assert_eq!(combined[0].1, params.lf_coefficient(a, b), "{ctx}");
                            if combined.len() > 1 {
                                assert!(combined[1].0.weight() > m.weight(), "{ctx}");
                            }
                        }
                        None => {
                            let w = a.weight() + b.weight();
                            assert!(combined.iter().all(|(m, _)| m.weight() > w), "{ctx}");
                        }
                    }
                }
            }
        }
    }
}

/// Every term of a product has May filtration at least the sum of those of the factors, which
/// makes the May-graded product well defined.
#[rstest]
#[case(2, SteenrodKind::Classical, 40)]
#[case(2, SteenrodKind::MotivicModTau, 40)]
#[case(3, SteenrodKind::Classical, 70)]
#[case(3, SteenrodKind::ReducedPowers, 70)]
#[case(5, SteenrodKind::Classical, 100)]
fn may_filtration_is_multiplicative(#[case] p: u32, #[case] kind: SteenrodKind, #[case] max: i32) {
    let algebra = milnor_algebra(p, kind);
    let params = Params::new(p, kind);
    for d1 in 0..=max {
        for d2 in 0..=max - d1 {
            for a in basis(&algebra, &params, d1) {
                for b in basis(&algebra, &params, d2) {
                    let w = params.may_filtration(a) + params.may_filtration(b);
                    params.mul(a, b, |m, _| {
                        assert!(
                            params.may_filtration(m) >= w,
                            "{} * {} has the term {}",
                            params.format(a),
                            params.format(b),
                            params.format(m)
                        );
                    });
                }
            }
        }
    }
}

#[rstest]
#[case(2, SteenrodKind::Classical, 60)]
#[case(2, SteenrodKind::MotivicModTau, 60)]
#[case(3, SteenrodKind::Classical, 100)]
#[case(5, SteenrodKind::Classical, 160)]
fn divisibility_ops(#[case] p: u32, #[case] kind: SteenrodKind, #[case] max: i32) {
    let algebra = milnor_algebra(p, kind);
    let params = Params::new(p, kind);
    let all: Vec<Mono> = (0..=max)
        .step_by(if p == 2 { 1 } else { 4 })
        .flat_map(|d| basis(&algebra, &params, d))
        .collect();
    let digits = |m: Mono| -> Vec<u32> {
        (0..params.positions().len())
            .map(|i| params.digit(m, i))
            .collect()
    };
    for &a in &all {
        for &b in &all {
            let (da, db) = (digits(a), digits(b));
            let (qa, qb) = (a.q_part(), b.q_part());
            let divides = qa & !qb == 0 && da.iter().zip(&db).all(|(x, y)| x <= y);
            assert_eq!(params.divides(a, b), divides);
            if divides {
                assert_eq!(params.lf_mul(params.quotient(b, a), a), Some(b));
            }
            let gcd = params.from_digits(
                qa & qb,
                da.iter()
                    .zip(&db)
                    .map(|(x, y)| *x.min(y))
                    .enumerate()
                    .filter(|&(_, d)| d > 0),
            );
            let lcm = params.from_digits(
                qa | qb,
                da.iter()
                    .zip(&db)
                    .map(|(x, y)| *x.max(y))
                    .enumerate()
                    .filter(|&(_, d)| d > 0),
            );
            assert_eq!(params.gcd(a, b), gcd);
            assert_eq!(params.lcm(a, b), lcm);
            assert_eq!(params.coprime(a, b), gcd.is_one());
            let no_carry = qa & qb == 0 && da.iter().zip(&db).all(|(x, y)| x + y < p);
            assert_eq!(params.lf_mul(a, b).is_some(), no_carry);
        }
    }
}

mod resolution {
    use std::sync::Arc;

    use algebra::SteenrodKind;

    use crate::{Bounds, FilterMode, FnFilter, Params, Resolution};

    fn sphere(p: u32, kind: SteenrodKind, s_max: u32, n_max: i32) -> Resolution {
        let mut res = Resolution::sphere(Arc::new(Params::new(p, kind)), Bounds { s_max, n_max });
        res.compute_through(n_max + s_max as i32);
        res
    }

    fn counts(res: &Resolution) -> Vec<Vec<i32>> {
        (0..=res.bounds().s_max)
            .map(|s| res.generators(s).iter().map(|g| g.degree).collect())
            .collect()
    }

    /// $\mathrm{Ext}^1$ is spanned by the $h_i$, and $\mathrm{Ext}^2$ by the products $h_i h_j$
    /// with $j \neq i + 1$.
    #[test]
    fn low_lines_at_two() {
        let res = sphere(2, SteenrodKind::Classical, 2, 40);
        let gens = counts(&res);
        assert_eq!(gens[1], vec![1, 2, 4, 8, 16, 32]);
        let mut expected: Vec<i32> = (0..6)
            .flat_map(|i| (i..6).map(move |j| (i, j)))
            .filter(|&(i, j)| j != i + 1)
            .map(|(i, j)| (1 << i) + (1 << j))
            .filter(|&t| t - 2 <= 40)
            .collect();
        expected.sort();
        assert_eq!(gens[2], expected);
    }

    /// Auditing a filter reduces every pair as usual and checks every prediction.
    #[test]
    fn audit_counts_outcomes() {
        let params = Arc::new(Params::new(2, SteenrodKind::Classical));
        let bounds = Bounds {
            s_max: 6,
            n_max: 20,
        };
        let mut audited = Resolution::sphere(Arc::clone(&params), bounds);
        audited.add_filter(Box::new(FnFilter::new("all", |_| true)), FilterMode::Audit);
        audited.add_filter(
            Box::new(FnFilter::new("none", |_| false)),
            FilterMode::Audit,
        );
        audited.compute_through(26);
        let plain = sphere(2, SteenrodKind::Classical, 6, 20);
        assert_eq!(counts(&audited), counts(&plain));

        let mut redundant = 0;
        let mut useful = 0;
        for (&(s, _), st) in audited.stats().steps() {
            if s >= 0 {
                let r = st.useless + st.boundaries + st.beyond;
                redundant += r;
                useful += st.reduced - r;
            }
        }
        let report = audited.filter_report();
        let total = |i: usize| {
            report.filters[i]
                .2
                .values()
                .fold((0, 0, 0), |(c, w, m), x| {
                    (c + x.correct, w + x.wrong, m + x.missed)
                })
        };
        assert_eq!(total(0), (redundant, useful, 0));
        assert_eq!(total(1), (0, 0, redundant));
    }

    /// A filter in force that predicts nothing changes nothing.
    #[test]
    fn enforce_without_predictions() {
        let params = Arc::new(Params::new(3, SteenrodKind::Classical));
        let bounds = Bounds {
            s_max: 5,
            n_max: 40,
        };
        let mut res = Resolution::sphere(params, bounds);
        res.add_filter(
            Box::new(FnFilter::new("none", |_| false)),
            FilterMode::Enforce,
        );
        res.compute_through(45);
        assert_eq!(
            counts(&res),
            counts(&sphere(3, SteenrodKind::Classical, 5, 40))
        );
        assert!(res.stats().total().useless > 0);
    }

    /// Lin's x2m pruning discards pairs, all of them redundant, without changing the result.
    #[rstest::rstest]
    #[case(2, SteenrodKind::Classical, 15, 40)]
    #[case(2, SteenrodKind::MotivicModTau, 12, 30)]
    #[case(3, SteenrodKind::Classical, 10, 80)]
    #[case(3, SteenrodKind::ReducedPowers, 8, 100)]
    #[case(5, SteenrodKind::Classical, 6, 150)]
    fn x2m(#[case] p: u32, #[case] kind: SteenrodKind, #[case] s_max: u32, #[case] n_max: i32) {
        let plain = sphere(p, kind, s_max, n_max);
        for mode in [FilterMode::Audit, FilterMode::Enforce] {
            let params = Arc::new(Params::new(p, kind));
            let mut res = Resolution::sphere(params, Bounds { s_max, n_max });
            res.enable_x2m(mode);
            res.compute_through(n_max + s_max as i32);
            assert_eq!(counts(&res), counts(&plain));
            let report = res.filter_report();
            let (discarded, correct, wrong) = report.filters[0]
                .2
                .values()
                .fold((0, 0, 0), |(d, c, w), x| {
                    (d + x.discarded, c + x.correct, w + x.wrong)
                });
            assert_eq!(wrong, 0);
            match mode {
                FilterMode::Audit => assert!(correct > 0),
                FilterMode::Enforce => assert!(discarded > 0),
            }
        }
    }
}
