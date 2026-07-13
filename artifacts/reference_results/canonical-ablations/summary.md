# Canonical HRM ablations — spatial-reconstruction

Seeds: 3

| Ablation | Mean utility effect | 95% CI | Wilcoxon p | Holm p | Diverged seeds | Cost-equal seeds |
|---|---:|---:|---:|---:|---:|---:|
| no_guidance | -5.41394e-05 | [-8.2699e-05, -1.3434e-05] | 0.25 | 1 | 3/3 | 3/3 |
| no_hrm_adaptation | -0.00139186 | [-0.00235644, 0.000372549] | 0.5 | 1 | 3/3 | 0/3 |
| fixed_topology | -0.00139171 | [-0.002356, 0.000372724] | 0.5 | 1 | 3/3 | 0/3 |
| fixed_geometry | 3.56506e-08 | [1.66269e-08, 7.34819e-08] | 0.25 | 1 | 3/3 | 0/3 |
| fixed_graph | -0.00139165 | [-0.00235591, 0.000372726] | 0.5 | 1 | 3/3 | 0/3 |
| no_hierarchy | -0.000947704 | [-0.00103445, -0.000783309] | 0.25 | 1 | 3/3 | 0/3 |
| memory_only | -0.0339751 | [-0.037921, -0.0290013] | 0.25 | 1 | 3/3 | 0/3 |
| fixed_allocation | -0.0421344 | [-0.0454962, -0.0368924] | 0.25 | 1 | 3/3 | 3/3 |
| no_memory | -0.0016815 | [-0.0169034, 0.00995727] | 1 | 1 | 3/3 | 0/3 |
| no_cognition_to_field | -2.164e-07 | [-3.61012e-07, -1.55232e-08] | 0.25 | 1 | 3/3 | 3/3 |
| no_field_to_cognition | -2.05242e-08 | [-3.72903e-08, 2.75163e-09] | 0.5 | 1 | 3/3 | 3/3 |
| no_regional_coupling | -0.000723323 | [-0.000895969, -0.000516482] | 0.25 | 1 | 3/3 | 3/3 |
| no_topology_repair | 0 | [0, 0] | 1 | 1 | 0/3 | 3/3 |
| no_adaptive_timestep | 0 | [0, 0] | 1 | 1 | 0/3 | 3/3 |

Positive utility means the full system had lower error. Statistical results do not imply support when activation, divergence, cost, or ledger gates fail.
