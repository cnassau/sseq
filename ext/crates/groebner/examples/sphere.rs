//! Resolve the sphere and print the number of generators in each bidegree.
//!
//! Usage: `sphere <p> <kind> <s_max> <n_max> [pair-log.jsonl]`, where kind is one of
//! `classical`, `motivic` and `reduced`. Set `X2M` to `audit` or `enforce` to turn on Lin's x2m
//! pruning.

use std::sync::Arc;

use algebra::SteenrodKind;
use groebner::{Bounds, FilterMode, PairLog, Params, Resolution};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p: u32 = args.get(1).map_or(2, |x| x.parse().unwrap());
    let kind = match args.get(2).map_or("classical", |x| x.as_str()) {
        "classical" => SteenrodKind::Classical,
        "motivic" => SteenrodKind::MotivicModTau,
        "reduced" => SteenrodKind::ReducedPowers,
        k => panic!("unknown kind {k}"),
    };
    let s_max: u32 = args.get(3).map_or(10, |x| x.parse().unwrap());
    let n_max: i32 = args.get(4).map_or(20, |x| x.parse().unwrap());

    let params = Arc::new(Params::new(p, kind));
    let mut res = Resolution::sphere(params, Bounds { s_max, n_max });
    match std::env::var("X2M").as_deref() {
        Ok("audit") => res.enable_x2m(FilterMode::Audit),
        Ok("enforce") => res.enable_x2m(FilterMode::Enforce),
        Ok(other) => panic!("unknown X2M mode {other}"),
        Err(_) => {}
    }
    if let Some(path) = args.get(5) {
        let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        res.set_observer(Some(Box::new(PairLog::new(file))));
    }
    let start = std::time::Instant::now();
    res.compute_through(n_max + s_max as i32);
    let elapsed = start.elapsed();

    for s in (0..=s_max).rev() {
        let mut row = String::new();
        for n in 0..=n_max {
            let count = res
                .generators(s)
                .iter()
                .filter(|g| g.degree - s as i32 == n)
                .count();
            row.push_str(&match count {
                0 => "  ".to_string(),
                c => format!("{c:>2}"),
            });
        }
        println!("{row}");
    }
    println!();
    print!("{}", res.stats());
    print!("{}", res.filter_report());
    println!("time: {elapsed:?}");
    let (mut x1, mut x2, mut n) = (0, 0, 0);
    for s in 0..s_max {
        for e in res.level(s).elements() {
            x1 += e.x1.num_terms();
            x2 += e.x2.num_terms();
            n += 1;
        }
    }
    println!("basis: {n} elements, {x1} + {x2} terms");
}
