# Hulk V7 external preflight

The one-click experiment trains the five Hulk development cases again in its
own run directory. External validation uses the frozen bundles from that same
run. The `--preflight-only` mode checks the declaration before training.

## Hulk manifest example

Create `external/` locally and save the following example as
`external/hulk_manifest.json`. Replace every placeholder using an independently
acquired capture before running the optional confirmation. The declaration stays
outside Git. Set `unseen_in_development` to true only when the
custodian can attest to that fact. No benchmark split or renamed development file
becomes an independent capture merely by appearing in this file.

```json
{
  "version": 1,
  "dataset": "CIC-IDS-2017",
  "target_class": "DoS Hulk",
  "attestation": {
    "custodian": "<actual capture custodian>",
    "unseen_in_development": false
  },
  "exact_overlap_policy": "reject",
  "captures": [
    {
      "capture_id": "<unique independent acquisition ID>",
      "prepared_csv": "<relative or absolute path to prepared capture CSV>",
      "sha256": "<SHA256 of prepared CSV>",
      "raw_files": [
        {
          "path": "<relative or absolute path to original capture file>",
          "sha256": "<SHA256 of original capture file>"
        }
      ]
    }
  ],
  "criteria": {
    "min_target_f1_effect_pp": 1.0,
    "max_far": 0.01,
    "max_far_increase": 0.001,
    "max_fnr_increase": 0.005,
    "min_target_groups": 5,
    "min_benign_groups": 5,
    "min_seeds": 5,
    "min_captures": 1,
    "min_target_f1_over_base_pp": 1.0,
    "max_macro_f1_loss_pp": 1.0
  }
}
```

The example criteria are starting values, not proof of practical usefulness.
Fix the intended criteria before viewing confirmation results. A single capture
supports claims limited to that acquisition; five model seeds are not five captures.
For a predeclared across-capture acceptance criterion, set `min_captures` to
at least two and add `min_capture_target_f1_ci95_lower_pp`. For example, zero
requires the capture-paired 95% interval to be entirely above zero. Each capture
used for this criterion needs both target and benign groups and every frozen
seed's paired F1. With one acquisition, the capture interval is undefined.
Paths are resolved relative to the manifest's directory. The prepared CSV must use
the frozen model's exact feature order and units, followed by `label,group_id`.
Labels follow `class_order`; group IDs should identify globally unique acquisition,
session or vehicle units. See `docs/external_confirmation.md` for the full contract.

`revision_external_preflight.preflight_hulk(source, manifest)` returns `ready`,
`pending` or `failed`. It validates the completed V12 attribution receipts, all
five V7 case result hashes and their frozen head, metadata and TabM checkpoint
hashes, the declaration schema and source hash exclusions, and whether the declared
prepared and raw files exist. Each seed's route, CAL partition, allocator and donor
are checked against its own evaluated record. It returns the expected CSV
columns and labels from the frozen receipt. It does not open capture contents,
compute predictions, fit thresholds or write a seal. Missing declarations, unfilled
templates and missing capture files are pending, not successful confirmation.

`run_verified_hulk(output, source, manifest)` proceeds only after a ready preflight.
The V7 workflow writes an immutable seal before the confirmation engine
reads capture rows or scores them. That engine then checks actual file hashes,
column order, numeric features, labels, group IDs and development overlap. A ready
preflight therefore does not establish that all capture-content checks will pass,
nor does a local hash establish physical acquisition independence or custody.

If a manifest was renamed before any seal or result exists, the wrapper can update
the same unsealed workflow after verifying unchanged source models. It archives the
previous workflow first. Once a seal or result exists, changing the manifest path is
rejected and the original result is retained. Matching completed runs are verified
and reused. If the final receipt is missing and the result tables are complete,
the program reruns sealed scoring in a temporary directory before accepting them.
Previous experiments and source models are preserved.
