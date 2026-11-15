# Cohere / Command-R pre-merge parity gate

This gate protects against adapter conversion errors that match HuggingFace on a
short prompt but accumulate over later positions.

## Required configuration

- Reference and bridge run in `float32` on CPU.
- The bridge uses eager attention.
- Weight processing is disabled for the raw-forward parity check.

## Required probes

Run the fixed prompt matrix in `tests/integration/model_bridge/test_cohere_adapter.py`:

1. `short-4`: the historical four-token smoke probe.
2. `long-32`: eight repetitions of that probe, to exercise accumulated drift.

For every probe, calculate the maximum absolute logit difference over every
batch item, position, and vocabulary value. The gate passes only when each
maximum is `<= 1e-4`.

## Review policy

A pass on `short-4` alone is not sufficient evidence for a Command-R adapter.
A failure of either matrix case is an adapter-correctness failure, not floating
point noise. Keep the adapter out of the release until the failing conversion
or forward-path behavior is fixed and the full matrix is green.

Record the command output and the per-case maxima in the PR before merge.
