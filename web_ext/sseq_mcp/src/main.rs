//! An MCP server that hands minimal resolutions over the Steenrod algebra to a SageMath session.

mod encode;
mod products;

use std::{
    collections::HashMap,
    fmt::Write,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use algebra::{Algebra, MilnorAlgebra};
use clap::Parser;
use ext::chain_complex::{ChainComplex, FreeChainComplex};
use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        ListResourcesResult, PaginatedRequestParams, ReadResourceRequestParams,
        ReadResourceResponse, ReadResourceResult, Resource, ResourceContents, ServerCapabilities,
        ServerConfig,
    },
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Deserialize;
use sseq::coordinates::Bidegree;

/// The resolutions this server computes.
type Res = ext::utils::QueryModuleResolution;

/// `sage/sseq.sage`, the Sage side of the resolutions.
const SAGE_MODULE: &str = include_str!("../sage/sseq.sage");
const SAGE_MODULE_URI: &str = "sseq://sage/sseq.sage";
const MODULES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../ext/steenrod_modules");

/// Largest `sseq_resolution_data` reply, in bytes. Bigger replies cost the model more to relay
/// than splitting the range does.
const MAX_DATA_LEN: usize = 200_000;

const INSTRUCTIONS: &str =
    "\
This server is the go-to point for Ext over the Steenrod algebra: minimal resolutions, Ext groups \
     and their generators, over the full mod p Steenrod algebra and over its sub-Hopf-algebras \
     such as A(n) and E(n) (given by profile functions). Use it for anything about Ext_A(M, F_p), \
     Adams E_2 pages or minimal resolutions rather than computing these by hand or in plain Sage.

The resolutions are meant to be studied in a SageMath session of the mcp-sage server, as a family \
     of CombinatorialFreeModules C_s over Sage's SteenrodAlgebra with algebra-linear \
     differentials d(s): C_s -> C_{s-1}. Sage code cannot reach this server, so you relay the \
     data:

1. mcp-sage start_sage_session; pass its workspace_token as `session` on every later call.
2. sseq_sage_module, then run the returned code with mcp-sage evaluate_sage (once per workspace).
3. sseq_resolve, then run the returned constructor line in the workspace.
4. sseq_resolution_data for the range you need, then run the returned code in the workspace.
5. If Sage raises SseqMissingData, make the sseq_resolution_data call its message names and retry.

In Sage, with R the resolution: R.ngens(s, t), R.gen(s, t, i), R.C(s), R.d(s), R.ext_dims(s_max, \
     t_max), R.degree_basis(s, t), R.check_d_squared(s_max, t_max). Ext^{s,t} has dimension \
     R.ngens(s, t) since the resolution is minimal. Generator counts are also available without \
     Sage from sseq_resolve, and products (the lines of an Adams chart, or products by any class \
     over the ground field) from sseq_products, and the chain maps realising products by a class \
     from sseq_chain_map.";

#[derive(Parser)]
struct Args {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8315")]
    bind: SocketAddr,
    /// Directory to keep resolutions in between runs, one subdirectory per resolution.
    #[arg(long)]
    save_dir: Option<PathBuf>,
}

/// A resolution, the ring its products come from, and the lock its computations take.
struct Entry {
    res: Arc<Res>,
    /// `Ext(k, k)` on `res`; its products by arbitrary classes are only meaningful when `res`
    /// resolves `k` (see [`products::is_unit`]).
    ext: ext::ext_algebra::ExtAlgebra<Res>,
    /// Held while extending `res`, so that two requests do not compute the same bidegrees at once.
    compute: Mutex<()>,
}

/// The resolutions computed so far, shared by every MCP session.
#[derive(Default)]
struct State {
    resolutions: Mutex<HashMap<String, Arc<Entry>>>,
    save_dir: Option<PathBuf>,
}

impl State {
    /// The resolution of the built-in module `module`, constructing it on first use.
    fn get_or_construct(&self, module: &str) -> anyhow::Result<Arc<Entry>> {
        if let Some(entry) = self.resolutions.lock().unwrap().get(module) {
            return Ok(Arc::clone(entry));
        }
        if !is_module_name(module) {
            anyhow::bail!(
                "{module:?} is not a built-in module name, optionally shifted as in C2[1]"
            );
        }
        let save_dir = self.save_dir.as_ref().map(|dir| dir.join(module));
        let res = ext::utils::construct((module, "milnor"), save_dir)?;
        let res = Arc::new(res);
        let entry = Arc::new(Entry {
            ext: ext::ext_algebra::ExtAlgebra::new(Arc::clone(&res)),
            res,
            compute: Mutex::new(()),
        });
        Ok(Arc::clone(
            self.resolutions
                .lock()
                .unwrap()
                .entry(module.to_string())
                .or_insert(entry),
        ))
    }

    /// The resolution called `id`, which must have been constructed already.
    fn get(&self, id: &str) -> Result<Arc<Entry>, ErrorData> {
        self.resolutions
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| {
                ErrorData::invalid_params(
                    format!("no resolution {id:?}; call sseq_resolve with module={id:?} first"),
                    None,
                )
            })
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ResolveRequest {
    /// A built-in module (see sseq_list_modules), optionally shifted, e.g. "S_2", "ko", "C2[1]".
    module: String,
    /// Resolve through this stem n = t - s.
    max_n: i32,
    /// Resolve through this homological degree s.
    max_s: i32,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct DataRequest {
    /// The resolution id returned by sseq_resolve.
    id: String,
    /// Smallest homological degree to send.
    s_min: i32,
    /// Largest homological degree to send.
    s_max: i32,
    /// Largest internal degree t to send; computed first if necessary.
    t_max: i32,
    /// Name of the SseqResolution variable in the Sage workspace.
    #[serde(default = "default_var")]
    var: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ProductsRequest {
    /// The resolution id returned by sseq_resolve.
    id: String,
    /// What to multiply by: a filtration-one class by name ("h_0", "h_1", "a_0", ...; sseq_resolve
    /// lists them), or, for resolutions of the ground field, a class given as "n s i" (generator i
    /// in stem n, filtration s) or as "n s" together with `coords`.
    by: String,
    /// Coordinates of the multiplier in the generator basis of its bidegree, when `by` is "n s".
    coords: Option<Vec<u32>>,
    /// Largest stem of the classes being multiplied.
    max_n: i32,
    /// Largest filtration of the classes being multiplied.
    max_s: i32,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ChainMapRequest {
    /// The resolution id returned by sseq_resolve; it must resolve the ground field.
    id: String,
    /// The class the chain map lifts: a filtration-one class by name ("h_0", ...), or "n s i"
    /// (generator i in stem n, filtration s), or "n s" together with `coords`.
    class: String,
    /// Coordinates of the class in the generator basis of its bidegree, when `class` is "n s".
    coords: Option<Vec<u32>>,
    /// Largest stem of the generators to evaluate the map on.
    max_n: i32,
    /// Largest filtration of the generators to evaluate the map on.
    max_s: i32,
}

fn default_var() -> String {
    "R".to_string()
}

/// The MCP service; one per session, all sharing a [`State`].
#[derive(Clone)]
struct SseqServer {
    state: Arc<State>,
    #[expect(dead_code, reason = "read by the code `tool_handler` generates")]
    tool_router: ToolRouter<Self>,
}

/// An internal error for the MCP client.
fn internal(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

/// Run `f` on a blocking thread, so long computations don't stall the async runtime.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> Result<T, ErrorData> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(internal)?
        .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))
}

#[tool_router]
impl SseqServer {
    fn new(state: Arc<State>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Sage code defining SseqResolution, the Sage wrapper for minimal \
                       resolutions over the Steenrod algebra. Run it once in a mcp-sage workspace \
                       (via evaluate_sage) before anything else from this server."
    )]
    fn sseq_sage_module(&self) -> String {
        SAGE_MODULE.to_string()
    }

    #[tool(
        description = "List the built-in Steenrod modules that can be resolved, with their prime \
                       and the sub-Hopf-algebra they are modules over."
    )]
    fn sseq_list_modules(&self) -> Result<String, ErrorData> {
        let mut names: Vec<_> = std::fs::read_dir(MODULES_DIR)
            .map_err(internal)?
            .filter_map(|e| {
                let path = e.ok()?.path();
                (path.extension()? == "json").then_some(path)
            })
            .collect();
        names.sort();
        let mut out = String::new();
        for path in names {
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).map_err(internal)?)
                    .map_err(internal)?;
            let name = path.file_stem().unwrap().to_string_lossy();
            let p = &json["p"];
            let over = match &json["algebra"]["profile"] {
                serde_json::Value::Null => "A".to_string(),
                profile => format!("profile {profile}"),
            };
            writeln!(out, "{name}: p = {p}, over {over}").unwrap();
        }
        Ok(out)
    }

    #[tool(
        description = "Compute the minimal resolution of a module over the Steenrod algebra \
                       through stem max_n and homological degree max_s. Returns the resolution \
                       id, the Sage line creating its SseqResolution, and the number of \
                       generators (= dim Ext^{s,t}) in each nonzero bidegree as `n s count` lines."
    )]
    async fn sseq_resolve(
        &self,
        Parameters(req): Parameters<ResolveRequest>,
    ) -> Result<String, ErrorData> {
        let state = Arc::clone(&self.state);
        blocking(move || {
            let entry = state.get_or_construct(&req.module)?;
            {
                let _guard = entry.compute.lock().unwrap();
                entry
                    .res
                    .compute_through_stem(Bidegree::n_s(req.max_n, req.max_s));
            }
            let res = &entry.res;
            let algebra = res.algebra();
            let algebra: &MilnorAlgebra = (&*algebra).try_into()?;
            let names: Vec<String> = products::named(res)
                .into_iter()
                .map(|(name, ..)| name)
                .collect();
            let mut out = format!(
                "id: {id}\nSage: R = sseq_resolution({id:?}, {p}, {algebra})\nNamed products: \
                 {names}{classes}\n\nn s count\n",
                id = req.module,
                p = algebra.prime(),
                algebra = encode::sage_algebra(algebra)?,
                names = names.join(", "),
                classes = if products::is_unit(res) {
                    "; products by any class are available too"
                } else {
                    ""
                },
            );
            for b in res.iter_stem() {
                let n = res.number_of_gens_in_bidegree(b);
                if n > 0 {
                    writeln!(out, "{} {} {n}", b.n(), b.s()).unwrap();
                }
            }
            Ok(out)
        })
        .await
    }

    #[tool(
        description = "Sage code adding the generators and differentials of C_s for s_min <= s <= \
                       s_max and internal degree t <= t_max to the SseqResolution `var` (default \
                       R). Run the result with mcp-sage evaluate_sage. Computes the range first \
                       if sseq_resolve has not."
    )]
    async fn sseq_resolution_data(
        &self,
        Parameters(req): Parameters<DataRequest>,
    ) -> Result<String, ErrorData> {
        if req.s_min < 0 || req.s_min > req.s_max {
            return Err(ErrorData::invalid_params("need 0 <= s_min <= s_max", None));
        }
        if !is_identifier(&req.var) {
            return Err(ErrorData::invalid_params(
                "var must be a Python identifier",
                None,
            ));
        }
        let entry = self.state.get(&req.id)?;
        blocking(move || {
            {
                let _guard = entry.compute.lock().unwrap();
                entry
                    .res
                    .compute_through_bidegree(Bidegree::s_t(req.s_max, req.t_max));
            }
            let code = encode::sage_data(&entry.res, &req.var, req.s_min, req.s_max, req.t_max);
            if code.len() > MAX_DATA_LEN {
                anyhow::bail!(
                    "the data for s in {}..={}, t <= {} is {} bytes, over the {MAX_DATA_LEN} byte \
                     limit; request fewer homological degrees at a time",
                    req.s_min,
                    req.s_max,
                    req.t_max,
                    code.len()
                );
            }
            Ok(code)
        })
        .await
    }

    #[tool(
        description = "Multiplicative structure of Ext: the products of a class with every \
                       generator in stems <= max_n and filtrations <= max_s, as lines `n s i -> \
                       n' s' : coordinates`. Use this for the h_0, h_1, h_2 (or a_0, h_0) lines \
                       of an Adams chart, or for products by any class of Ext of the ground \
                       field. Products are determined up to sign. Computes further first if \
                       needed."
    )]
    async fn sseq_products(
        &self,
        Parameters(req): Parameters<ProductsRequest>,
    ) -> Result<String, ErrorData> {
        let entry = self.state.get(&req.id)?;
        blocking(move || {
            let res = &entry.res;
            let x = products::parse(res, &req.by, req.coords.as_deref())?;
            let shift = x.degree();
            // Products are computed under the lock, since a class product extends chain maps
            // over the whole computed resolution.
            let _guard = entry.compute.lock().unwrap();
            res.compute_through_stem(Bidegree::n_s(req.max_n + shift.n(), req.max_s + shift.s()));
            let out = products::table(res, &entry.ext, &x, req.max_n, req.max_s)?;
            if out.len() > MAX_DATA_LEN {
                anyhow::bail!(
                    "the products are {} bytes, over the {MAX_DATA_LEN} byte limit; ask for a \
                     smaller range",
                    out.len()
                );
            }
            Ok(out)
        })
        .await
    }

    #[tool(
        description = "The chain map gamma: C_s -> C_(s - s0) of the minimal resolution that \
                       lifts a class of Ext in filtration s0 (multiplication by that class at \
                       chain level), evaluated on every generator in stems <= max_n and \
                       filtrations <= max_s. Lines `n s generator : gamma(generator)`, the values \
                       written in the Milnor basis. Only for resolutions of the ground field \
                       (S_2, S_3, tmf2, ...). Computes further first if needed."
    )]
    async fn sseq_chain_map(
        &self,
        Parameters(req): Parameters<ChainMapRequest>,
    ) -> Result<String, ErrorData> {
        let entry = self.state.get(&req.id)?;
        blocking(move || {
            let res = &entry.res;
            anyhow::ensure!(
                products::is_unit(res),
                "chain maps need a resolution of the ground field (such as S_2 or tmf2)"
            );
            // Chain maps are built under the lock, since they extend over the computed resolution.
            let _guard = entry.compute.lock().unwrap();
            res.compute_through_stem(Bidegree::n_s(req.max_n, req.max_s));
            let x = products::parse(res, &req.class, req.coords.as_deref())?.to_class(res)?;
            let out = products::chain_map_table(res, &entry.ext, &x, req.max_n, req.max_s);
            if out.len() > MAX_DATA_LEN {
                anyhow::bail!(
                    "the table is {} bytes, over the {MAX_DATA_LEN} byte limit; ask for a smaller \
                     range",
                    out.len()
                );
            }
            Ok(out)
        })
        .await
    }
}

/// Whether `s` has the shape of a module name, optionally shifted.
///
/// The name becomes a file name, both of the module spec and of its save directory, so this keeps
/// requests inside those directories.
fn is_module_name(s: &str) -> bool {
    let (name, shift) = match s.split_once('[') {
        Some((name, shift)) => (name, Some(shift)),
        None => (s, None),
    };
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && shift.is_none_or(|shift| {
            shift
                .strip_suffix(']')
                .is_some_and(|n| n.trim_start_matches('-').parse::<u32>().is_ok())
        })
}

/// Whether `s` is a plain Python identifier, so that it is safe to splice into Sage code.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[tool_handler]
impl ServerHandler for SseqServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(INSTRUCTIONS)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![
            Resource::new(SAGE_MODULE_URI, "sseq.sage").with_description(
                "The Sage wrapper for minimal resolutions (same as sseq_sage_module)",
            ),
        ]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        if request.uri != SAGE_MODULE_URI {
            return Err(ErrorData::resource_not_found(
                format!("no resource {}", request.uri),
                None,
            ));
        }
        Ok(
            ReadResourceResult::new(vec![ResourceContents::text(SAGE_MODULE, SAGE_MODULE_URI)])
                .into(),
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let state = Arc::new(State {
        save_dir: args.save_dir,
        ..Default::default()
    });
    let service = StreamableHttpService::new(
        move || Ok(SseqServer::new(Arc::clone(&state))),
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!("sseq MCP server listening on http://{}/mcp", args.bind);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_names() {
        for good in ["S_2", "A-mod-Sq1-Sq2", "C2[1]", "RP_-4_inf[-3]"] {
            assert!(is_module_name(good), "{good}");
        }
        for bad in [
            "",
            "../S_2",
            "/etc/x",
            "S_2[",
            "S_2[x]",
            "S_2[1]/..",
            "a.json",
        ] {
            assert!(!is_module_name(bad), "{bad}");
        }
    }

    #[test]
    fn identifiers() {
        assert!(is_identifier("R") && is_identifier("_res2"));
        assert!(!is_identifier("2R") && !is_identifier("x;os") && !is_identifier(""));
    }
}
