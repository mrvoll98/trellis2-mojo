# ADR 0001: Hybrid vs Full Port

**Decision:** Hybrid (Python glue + Mojo kernels) for v0.1.

**Rationale:** Faster validation, interop for deps, focus on perf wins where they matter (sampling loops, sparse ops).

Full pure later.
