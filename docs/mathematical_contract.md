# Mathematical contract

Let the committed state be

\[
X_t=(\Phi_t,G_t,T_t,M_t,C_t,H_t,B_t).
\]

Every ordinary mechanism \(k\) reads the same immutable \(X_t\) and proposes a block-sparse
increment \(\Delta^{(k)}_t\). Declared interactions \(q\), intentional seeded stochastic terms,
event effects, and corrections remain separate:

\[
X_{t+1}-X_t =
\sum_k \Delta^{(k)}_t +
\sum_q \Delta^{(q)}_t +
\Delta^{(\mathrm{rng})}_t +
\Delta^{(\mathrm{event})}_t +
\Delta^{(\mathrm{correction})}_t + r_t.
\]

Production commits require both the absolute and relative norm of \(r_t\) to be no larger than the
configured tolerance. The independent ledger verifier reconstructs the right-hand side from stored
arrays and fails if they are missing or altered.

## Reference field equation

The explicit-Euler reference step separates the following attributable terms:

\[
\dot\Phi = -\gamma\Phi + I - D L_{T,G}\Phi + R(\Phi) + K_H + K_C + K_M + U.
\]

The default local reaction is

\[
R(\phi)=a\phi-b\phi^3-c\phi\lVert\phi\rVert^2,
\]

with an optional bounded tanh form. The graph Laplacian is positive semidefinite under the
combinatorial or symmetric convention, so diffusion uses `-D * L * Phi`. The timestep is reduced
deterministically when the spectral stability bound is below the configured step.

## Hybrid events

Events are not derivatives. An event has an ID, type, candidate basis, affected entities, score,
predicted utility, cost, risk, validation result, ordering key, transport metadata, and applied
effect. Edge growth, pruning, rewiring, node/channel activation and deactivation, memory write and
eviction, and repairs are represented explicitly. Edge flip is declared as optional and is rejected
by this reference implementation rather than accepted as a no-op.

## Scientific interpretation

An implemented term is not presumed useful. Evidence is gated in order: execution, nonzero
activation, matched-trajectory divergence, then outcome utility at reported cost. Negative utility
is a valid experimental result.
