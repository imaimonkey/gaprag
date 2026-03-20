# GapVerify TODO

## Near-term

- Add verification-focused analysis plots:
  - label calibration
  - verdict confidence
  - changed output vs changed verdict
- Add per-label gap statistics for FEVER-style tasks.

## Mid-term

- Add `SciFact` or `Climate-FEVER` for domain transfer.
- Add verification-specific injector variants.
- Add benchmark registry / result table generator for FEVER-family suites.
- Reframe memory experiments as diagnostic appendix runs.

## Engineering

- Add unit tests for dataset adapters and verification parsing.
- Add CI for lint + py_compile + smoke checks.
- Keep `run_experiments.sh` presets aligned with the verification-first research framing.
