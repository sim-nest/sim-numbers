# sim-numbers

A tower of number types -- exact fractions, arbitrary-precision integers,
complex, symbolic algebra, and n-dimensional tensors -- that add up together
under one shared arithmetic surface for your Rust runtime.

SIM ships a CLI binary `sim` (`cargo install sim-run`); for the full guided
walkthrough of the constellation see `sim-say`. The crates below are the number
**libraries** those runtimes load.

## Example

Add one of the number-domain crates and use it directly. The rational domain
reduces exact fractions to lowest terms as it parses them:

```bash
cargo add sim-lib-numbers-rational num-bigint
```

```rust
use num_bigint::BigInt;
use sim_lib_numbers_rational::parse_rational_parts;

// "6/8" parses and reduces to the normalized pair 3/4.
assert_eq!(
    parse_rational_parts("6/8"),
    Some((BigInt::from(3), BigInt::from(4)))
);
// A zero denominator is rejected.
assert_eq!(parse_rational_parts("1/0"), None);
```

(From the `parse_rational_parts` doctest in
`crates/sim-lib-numbers-rational/src/implementation/ops.rs:23`.)

## How it works

`sim-numbers` is the number surface of the SIM constellation. SIM is an
expandable Rust runtime built around a small protocol kernel plus a large set
of loadable libraries: the kernel defines contracts, libraries provide
behavior. Number representation and arithmetic are exactly this kind of
behavior -- the kernel defines `NumberDomain`/`NumberValue` and the value/expr
contracts but no concrete number domain. This repository supplies the domains.

The crates here register concrete number domains over the kernel contracts:
integer, floating-point, rational, bigint, complex, boolean, and exotic
scalars; cross-domain arithmetic over a promotion lattice; a computer-algebra
layer; numeric analysis backends; and an n-dimensional tensor domain with
per-element-type specializations. Each domain is a loadable library, and the
prelude installs the standard set in one call.

## Crates

### Substrate and assembly

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-core` | Shared substrate for scalar domains: the number-value shape and browse table, the shared number-literal shape and class, the scalar-domain spec, literal matcher, and op-loop installer, and the canonical number-domain symbol registry and promotion-lattice documentation. |
| `sim-lib-numbers-prelude` | Umbrella library: `NumbersPreludeLib` installs the standard number domains and tensor backends into a runtime in one call. |
| `sim-lib-numbers-codec` | The number-literal codec surface: helpers that build the `numeric-plugin-v1` descriptor symbols and table values advertising a numeric codec or method provider to the runtime. |

### Scalar number domains

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-bool` | The `numbers/bool` domain: boolean literals and values as the base of the promotion lattice, with edges widening into the integer and float domains. |
| `sim-lib-numbers-i64` | The `numbers/i64` domain: 64-bit signed-integer literals and values, their scalar arithmetic, and promotion into `f64` and `rational`. |
| `sim-lib-numbers-fixed` | The fixed-width integer domains (`numbers/i8` .. `numbers/i128`, `numbers/u8` .. `numbers/u128`): their literals, values, and widening promotion through the signed and unsigned integer lattice. |
| `sim-lib-numbers-bigint` | The `numbers/bigint` domain: arbitrary-precision integer literals and values, their exact arithmetic, and promotion into `rational`. |
| `sim-lib-numbers-rational` | The `numbers/rational` domain: exact rationals over bigint numerator/denominator pairs, their reduced arithmetic, and promotion edges to and from the integer and `f64` domains. |
| `sim-lib-numbers-float` | The `numbers/f32` domain: single-precision floating-point literals and values, their scalar arithmetic, and promotion into `f64`. |
| `sim-lib-numbers-f64` | The `numbers/f64` domain: double-precision floating-point literals and values, their scalar arithmetic, and promotion into `complex`. |
| `sim-lib-numbers-complex` | The `numbers/complex` domain: complex literals and values, their arithmetic, and the promotion edges from `i64`, `f64`, and `rational` into the complex sink of the scalar lattice. |
| `sim-lib-numbers-exotic` | Exotic number domains: the lazy continued-fraction domain (`numbers/cf`), infinite-precision reals carried as continued-fraction coefficient streams, their builtin constants, and the `as-f64` truncation function. |

### Cross-domain arithmetic

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-arith` | Cross-domain arithmetic: the `math/add`, `math/sub`, `math/mul`, `math/div`, and reduction entry points that coerce mixed-domain operands through the promotion lattice and route symbolic inputs to the CAS when it is loaded. |

### Computer algebra and function domains

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-cas` | The `numbers/cas` domain: the core computer-algebra layer -- the symbolic expression tree `CasExpr`, its value citizen, the conversions to and from surface `Expr`/`Value`, and the `cas/var` and `cas/simplify` functions other CAS crates build on. |
| `sim-lib-numbers-cas-eval` | Evaluation of `numbers/cas` symbolic expressions: the `eval-cas` function that walks a `CasExpr` against an environment in numeric or symbolic mode, and the surface `Expr`/`CasExpr` bridge it uses. |
| `sim-lib-numbers-cas-diff` | Symbolic differentiation and integration over the `numbers/cas` tree: the `diff` and `integrate-sym` functions plus an extensible registry of per-operator differentiation rules. |
| `sim-lib-numbers-func` | The function number domain: callable function values built over CAS or native bodies, with `fn`, `call`, and `grad` operations for the `Func` domain. |

### Numeric analysis

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-numeric` | The numeric evaluation surface: the `numeric` domain exposing `numeric-diff`, `integrate`, and `ode-solve` over a registry of pluggable differentiator, quadrature, and ODE-solver backends. |
| `sim-lib-numbers-quad` | Quadrature and finite-difference backends for the numeric domain: fixed and adaptive integration rules plus finite-difference differentiators, packaged as a numeric plugin library. |
| `sim-lib-numbers-rk` | Runge-Kutta ODE integrators for the numeric domain: fixed-step and adaptive solver backends registered as numeric `ode-solve` plugins. |
| `sim-lib-numbers-ad` | Automatic differentiation primitives: forward-mode dual numbers, a reverse-mode evaluation tape, and the `Scalarish` numeric trait they share. |
| `sim-lib-numbers-stats` | Probability, descriptive statistics, and fairness metric helpers for `f64` number-domain data. |

### Tensor domain and specializations

| Crate | Role |
| --- | --- |
| `sim-lib-numbers-tensor` | The n-dimensional tensor number domain: the uniform `Tensor` value, its domain registration and constructors (`tensor`, `vec`, `mat`, ...), and the `SpecTensor` interface that specialized element-type backends plug into. |
| `sim-lib-numbers-tensor-f64` | f64 tensor specialization: a contiguous `f64` element type and its `SpecTensor` backend with native element-wise math. |
| `sim-lib-numbers-tensor-i64` | i64 tensor specialization: a contiguous `i64` element type and its `SpecTensor` backend, with overflow-checked operations that fall back to the bigint domain. |
| `sim-lib-numbers-tensor-rat64` | Rational-i64 tensor specialization: a normalized `(numerator, denominator)` i64-pair element type and its `SpecTensor` backend for the rational tensor domain. |
| `sim-lib-numbers-tensor-cmplxf` | Complex-float tensor specialization: a `(real, imag)` f64-pair element type and its `SpecTensor` backend for the complex tensor domain. |
| `sim-lib-numbers-tensor-bit` | Bit-tensor specialization: a packed-word boolean element type and its `SpecTensor` backend, with bitwise operations over the tensor domain. |
| `sim-lib-numbers-tensor-bcast` | Tensor broadcasting specialization: element-wise binary and unary tensor operations with NumPy-style shape broadcasting and promotion rules. |
| `sim-lib-numbers-tensor-linalg` | Linear-algebra operations over the tensor domain: `dot`, `matmul`, `cross`, `transpose`, `det`, `inv`, `trace`, `norm`, and the `eye`/`zeros`/`ones` constructors. |

## Number domains and the promotion lattice

A scalar number domain is a loadable library that registers a domain symbol
(for example `numbers/i64`), a literal matcher, the domain's values, and its
arithmetic op-loop over the shared substrate in `sim-lib-numbers-core`. Domains
declare promotion edges into wider domains, forming a lattice that runs from
`bool` through the integer and floating-point domains up into `complex`.
Cross-domain operators in `sim-lib-numbers-arith` coerce mixed-domain operands
along these edges before dispatching, and route symbolic operands to the CAS
domain when it is loaded.

The tensor domain follows the same shape one level up: `sim-lib-numbers-tensor`
defines the uniform `Tensor` value and the `SpecTensor` interface, and each
specialization crate registers a fast element-type backend (`f64`, `i64`,
`rational`, `complex`, `bool`) for it, falling back to the uniform
representation where no specialization applies.

## Validation

Run validation from this repository:

```bash
cargo fmt --all --check && cargo run -p xtask -- check-file-sizes && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo doc --workspace --no-deps
cargo run -p xtask -- simdoc --check
```

## Documentation Lanes

`cargo run -p xtask -- simdoc` builds the public documentation lanes:

- API docs: `target/doc/`
- Agent cards: `docs/agents/cards.jsonl` and `docs/agents/card-index.json`
- Human docs: `docs/humans/`
- Diagrams: `docs/diagrams/src/` and `docs/diagrams/generated/`

The same command writes split contract files under `docs/generated/`. Everything
under `docs/` is generated; do not hand-edit it.

### Rustdoc conventions

Public API documentation in `src/` follows one house style:

- Every public item opens with a one-line summary sentence, then context.
- The kernel defines `NumberDomain`/`NumberValue` and the value/expr contracts
  but no concrete number behavior; these crates supply the domains, arithmetic,
  CAS, tensors, and analysis. Each item is framed by its number-domain role.
- The first-reach types carry a `# Examples` doctest that compiles and passes.
- Cross-reference with intra-doc links, and link back to this README rather than
  restating it.

The public API is documentation-gated: each crate's `lib.rs` denies
`missing_docs`, so every public item, field, and variant must be documented for
the crate to build.

### Examples and recipes

Every crate ships runnable recipes under its `recipes/` directory; those plus the
crates' rustdoc doctests are the worked examples for the number domains.
