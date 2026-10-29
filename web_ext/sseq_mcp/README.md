# sseq_mcp

An [MCP](https://modelcontextprotocol.io) server that computes minimal resolutions over the
Steenrod algebra and hands them to a SageMath session running in a separate mcp-sage server. In
Sage a resolution is an `SseqResolution`: a family of graded free modules `C_s` over Sage's
`SteenrodAlgebra` (or a sub-Hopf-algebra given by a profile, such as A(1)), with the minimal
generators as basis and algebra-linear differentials `d(s): C_s -> C_{s-1}` preserving degrees.
An element `a * x[(t, i)]` has degree `deg(a) + t`, which `degree()`, `homogeneous_component(n)`
and the other graded-module methods take into account.

## Running

```sh
cargo run --release -- --bind 127.0.0.1:8315 [--save-dir DIR]
claude mcp add --transport http sseq http://127.0.0.1:8315/mcp
```

`--save-dir` keeps each resolution in `DIR/<module>` in the format of `ext/SAVE-FORMAT.md`, so a
restarted server picks up where it left off.

## How the two servers cooperate

mcp-sage runs caller code in a sandbox without imports or network access, so Sage cannot call this
server. The model relays instead:

| Step | Server | Tool | Then |
| --- | --- | --- | --- |
| 1 | mcp-sage | `start_sage_session` | use the returned `workspace_token` as `session` |
| 2 | sseq | `sseq_sage_module` | run the returned code with `evaluate_sage` |
| 3 | sseq | `sseq_resolve` | run the returned `R = sseq_resolution(...)` line |
| 4 | sseq | `sseq_resolution_data` | run the returned `R.add_data(...)` line |

Asking `R` about a bidegree it has no data for raises `SseqMissingData`, whose message names the
`sseq_resolution_data` call to make. `sseq_list_modules` lists the modules in
`ext/steenrod_modules`.

`sseq_products` gives the multiplicative structure without going through Sage: the products of a
class with every generator in a range, as `n s i -> n' s' : coordinates` lines. The class is a
named filtration-one class (`h_0`, `h_1`, ... at p = 2, `a_0`, `h_0` at odd primes), read off the
differentials, or, for a resolution of the ground field, any class of Ext, through chain maps.
Products are determined up to sign.

`sseq_chain_map` shows the chain level behind a product: the chain map `C_s -> C_(s - s0)` lifting
a class of filtration `s0`, evaluated on every generator in a range, with values in the Milnor
basis. It needs a resolution of the ground field.

`sseq_lift` solves lifting problems `d(y) = z` for a list of cycles `z`, written in the notation the
other tools print, so their output can be fed back in. Cycles of one bidegree are lifted together;
a cycle without a lift gets the reason (`d(z) != 0`, or nonzero under the augmentation in `C_0`).

`sseq_steenrod_module` builds any module known to sseq in Sage, as a `SseqSteenrodModule`: a graded
vector space over `F_p` with the action of every Milnor basis element through a degree `t_max`, so
that `a * x` works for `a` in `M.algebra()`. Finite modules come whole; infinite ones are truncated
at `t_max`. Specs that are the cofiber of a map (`C4`, `C9`, ...) are chain complexes rather than
modules and are refused.

The Sage side lives in [`sage/sseq.sage`](sage/sseq.sage) and is compiled into the server. Because
of the sandbox it uses no imports, no dunder names and none of Sage's persistence functions; keep
it that way when editing it.

## Example

```python
R.ngens(2, 8)                      # 1: h2^2
R.d(2)(R.gen(2, 8, 0))             # Sq(6)*x1[(2, 0)] + (Sq(1,1)+Sq(4))*x1[(4, 0)]
R.check_d_squared(4, 12)           # []
R.ext_dims(4, 12)                  # {(n, s): dim Ext^{s, n+s}, ...}
R.degree_basis(2, 6)               # F_2-basis of (C_2)_6
```
