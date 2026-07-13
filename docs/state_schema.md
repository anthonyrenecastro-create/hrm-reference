# State schema

The authoritative state is serialized as schema `hrm-state-1.0` inside a compressed NPZ. Arrays are
JSON encoded to make restoration auditable rather than dependent on pickling.

| Block | Principal arrays | Structural metadata | Derived diagnostics |
|---|---|---|---|
| Field | `values[n,d]`, active masks, optional target | version | energy, local coherence, variance, collapse indicator |
| Geometry | `coordinates[n,p]`, scale, metric, anchors, bounds | version | curvature, stress, accumulated distortion exposure |
| Topology | sparse `weights[n,n]` | canonical edges/simplices, edge ages, event history, topology/operator versions | degree, components, validity |
| Memory | working vector, keys, values, strengths, ages, occupied mask | provenance, read/write/eviction counters, version | interference, occupancy |
| Cognition | latent, prediction, task prediction, residual, uncertainty, contexts | version | none hidden |
| Hierarchy | assignments, coarse state, restriction, prolongation, routing | feedback gain, version | aggregation loss |
| Resources | budget, use, allocations, widths, priorities, allowances | version | cumulative operations and costs |

Validation checks carrier shapes, finiteness, nonnegative quantities, canonical symmetric topology,
edge/weight support agreement, resource widths, hierarchy operator shapes, and configuration
compatibility. Clone operations deep-copy all state and restore writable array flags.

Diagnostic caches never authorize hidden dynamics. They are deterministically refreshed once after
the ledger-observed numerical delta is formed. State digests include the complete structured state,
including versions and run metadata; the numerical ledger projection deliberately excludes derived
caches and histories.
