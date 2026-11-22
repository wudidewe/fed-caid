# Changelog

## Unreleased — consolidated review workflow

- Run dual head TabM with client scalar CAL as the primary benchmark on four
  datasets. Report matched local and federated subtype controls, fixed 0.70
  threshold sensitivity and the original TabM reference separately.
- Use root `main.py` for full review, synthetic check, focused experiments,
  continuation and report export.
- Run a fresh review when all inputs are present and list missing inputs
  otherwise. Use `--demo` or `--smoke` for a synthetic check.
- Bind frozen receiver artifacts and score routing to the original run. Verify
  unsigned result directories by running the sealed scoring again.
- Preserve completion status when rebuilding reports. Reject duplicate summary
  rows and mixed evaluation roles before publication.
- Serialize local collector requests and bound send and receive operations.
  Reject incomplete declared LAN sessions.
- Record donor qualification on the CAL split the case actually uses.
- Retain reference candidate random streams by default for NB15. A separate
  exploratory run can select independent client streams.
- Include both NB15 detector weight and CAL threshold studies in the full review.
- Retain original evidence receipts and write the new workflow under `outputs/fedcaid_review`.
- Remove retired standalone pilot launchers and use the shared grouped
  validation configuration for the main and reference experiments.
- Check the bundled reports, synthetic CPU experiment and installed wheel from the current source.

## Unreleased - experimental revision v11.1

- Add a pure-CAL operating-point selector: candidate FAR grid, group-disjoint
  threshold-fit/operating-select subsets, receiver and pooled FAR, FNR increase
  against the 1% CAL reference, support minima and attack coverage.
- Never export a frozen receiver when no candidate is feasible; count
  `no_feasible_operating_point` explicitly in the overall assessment.
- Record the selected threshold, reference threshold, CAL group hash and
  selection status in frozen receipts; verify them during sealing.
- Keep a separate threshold-fit benign support floor; below it the selector
  reports `no_feasible_operating_point` instead of pooling heterogeneous
  client scores into an over-permissive threshold.

## Unreleased — experimental revision v7

- Split CAL by exact model-input groups across all simulated clients.
- Replay CAL-fit policies on verified probability caches.
- Guard empirical alarm thresholds and preserve the residual CAL FAR budget.
- Separate CAL-only and TRAIN+CAL absence, and report paired target and receiver outcomes.

## Unreleased — experimental revision v6

The v6 experiment entry is an experiment revision, not an assertion that a new
public software release or archive has been published. The package retains its
existing 1.0.0 metadata until the authors choose and archive a release version.

- Add controlled class-absence experiments and a matched conditional score
  adapter candidate, with explicit development/testing scope and communication
  accounting. A positive detection gain is not assumed.
- Add bounded two-collector TCP protocol validation using synthetic updates,
  framed application-byte counts and plaintext-equivalence checks. Default runs
  are same-host checks; no independent-machine or production-security claim is
  inferred from them.
- Extend regression tests for duplicate/stale submissions, invalid vectors,
  worker failures, timeout handling and fixed-point overflow contracts.
- Extend the CPU CI workflow with v6 smoke and installed-wheel resource/console
  checks outside the checkout. Public CI success is established only by actual
  successful workflow runs, not by the presence of this file.
- Add citation metadata without an invented DOI, release date or archive link.

## Experimental revision v5 — 2026-11-12

- Reuse the completed 110-case main matrix, regenerate its report, and declare
  55 supplementary training groups including both CAN quantization protocols.
- Distinguish completed execution from unavailable or low-support endpoints.
- Preserve negative contrasts, donor costs and historical evidence hashes.

## Experimental revision v4 — 2026-11-11

- Use versioned train-edge clipping for PLE stability, forward-purged temporal
  roles, matched transfer controls and declared gate sensitivity.
- Record experiment coverage independently of efficacy and manual review tasks.
