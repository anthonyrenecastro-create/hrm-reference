# Mechanisms

| Mechanism | Primary write | Reference rule | Main control |
|---|---|---|---|
| Passive | field | linear decay | `passive_decay` |
| Input | field | typed external drive | `input_gain` |
| Diffusion | field | mask-aware sparse `-D L Phi` | `diffusion`, Laplacian kind |
| Reaction | field | cubic competition or tanh | reaction coefficients |
| Regional | field | region residual coupling | `regional_gain` |
| Memory | memory plus typed slot events | associative retrieval, decay, gated writes | threshold, temperature, allowance |
| Cognition | cognition | bounded predictive/residual dynamics | `cognition_gain` |
| Guidance | field/cognition | clipped uncertainty/residual precursor | authorization, gain, bound |
| Hierarchy bottom-up | hierarchy | restriction and aggregation | bottom-up gain |
| Hierarchy top-down | field | prolongated coarse feedback | feedback gain/bound |
| Geometry | coordinates | stress forces plus anchor penalty | gain and displacement bound |
| Topology | typed events | utility-cost-risk growth, prune, rewire | thresholds and constraints |
| Allocation | resources plus channel events | importance-based active widths | budget and width bounds |

Named couplings—field→cognition, cognition→field, memory→cognition, memory→field, and
hierarchy→cognition—are separate proposals and can be ablated independently. The
residual–uncertainty interaction is likewise explicit.

Activation means a nonzero contribution or at least one event candidate. Execution without
activation is reported honestly. Topology receives no credit for diffusion over existing edges;
its utility is assessed through future divergence from matched fixed-topology/fixed-graph runs.
