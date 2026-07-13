# Mapping: pipelines/samplers/flow_euler.py

Euler sampler for flow matching.

Key funcs: _v_to_xstart_eps, sample_once, sample (with CFG mixins).

Pure math on (sparse) tensors.

**Good first port target:** Small, testable. Port math to Mojo, keep loop in py initially.

Reuse in Phase 2.
