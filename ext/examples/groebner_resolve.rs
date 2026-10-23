//! Resolves a module with Lin's Gröbner basis algorithm and prints the Ext chart together with
//! statistics on the critical pairs.
//!
//! The algebra is the classical Steenrod algebra unless the environment variable `GROEBNER_KIND`
//! is `motivic` (for $A^{\mathbb{C}}/\tau$ at $p = 2$) or `reduced` (for the reduced powers at an
//! odd prime). `GROEBNER_X2M`, `audit` or `enforce`, turns on Lin's x2m pruning. If
//! `GROEBNER_PAIR_LOG` names a file, every reduction is logged there as a line of JSON.

use ext::chain_complex::FreeChainComplex;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let kind = match std::env::var("GROEBNER_KIND").as_deref() {
        Ok("motivic") => Some(algebra::SteenrodKind::MotivicModTau),
        Ok("reduced") => Some(algebra::SteenrodKind::ReducedPowers),
        _ => None,
    };
    let module: String = query::with_default("Module", "S_2", |s| {
        Result::<_, std::convert::Infallible>::Ok(s.to_owned())
    });
    let res = ext::utils::construct_groebner(module.as_str(), kind)?;
    if let Ok(path) = std::env::var("GROEBNER_PAIR_LOG") {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        res.set_observer(Some(Box::new(groebner::PairLog::new(file))));
    }

    let max = Bidegree::n_s(
        query::with_default("Max n", "30", str::parse),
        query::with_default("Max s", "7", str::parse),
    );
    let start = std::time::Instant::now();
    res.compute_through_stem(max);
    let elapsed = start.elapsed();

    println!("{}", res.graded_dimension_string());
    if let Some(stats) = res.stats() {
        print!("{stats}");
    }
    if let Some(report) = res.filter_report() {
        print!("{report}");
    }
    eprintln!("time: {elapsed:?}");
    Ok(())
}
