# Frozen receiver evaluation

Use this workflow with independently acquired traffic after the development candidate, calibration policy, primary target and analysis criteria have been fixed. The software records the declarations and validates file identity, schema and previously seen exact inputs. The capture custodian is responsible for choosing truly unused acquisitions and providing meaningful session, vehicle or capture-block IDs.

## Data supplied by the capture custodian

- Original files for each new acquisition, with SHA256 values. Supply each acquisition once.
- One prepared UTF-8 CSV per capture. Use the exact model feature order from the frozen head receipt `feature_order`, followed by `label,group_id`. Features are pre-model-transform numeric values, in the same units and with the same derived-feature definitions as development. Empty values and NaN denote missing numeric measurements. Labels match the receipt `class_order` exactly, including `BENIGN`. Use a stable, globally namespaced identifier for each independent unit, such as `capture202611-a/session-014`. All rows from that unit share the same ID.
- A manifest with one dataset and one prespecified primary target. Relative file paths are resolved against the manifest directory. Get SHA256 with PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <file>`.

Example manifest. Replace the placeholder source paths, hashes and custodian name.

```json
{
  "version": 1,
  "dataset": "CIC-IDS-2017",
  "target_class": "DoS Hulk",
  "attestation": {
    "custodian": "<name>",
    "unseen_in_development": true
  },
  "exact_overlap_policy": "reject",
  "captures": [{
    "capture_id": "capture202611-a",
    "prepared_csv": "external/capture-a-prepared.csv",
    "sha256": "<prepared SHA256>",
    "raw_files": [{
      "path": "external/capture-a-original.csv",
      "sha256": "<original SHA256>"
    }]
  }],
  "criteria": {
    "min_target_f1_effect_pp": 1.0,
    "max_far": 0.01,
    "max_far_increase": 0.001,
    "max_fnr_increase": 0.005,
    "min_target_groups": 5,
    "min_benign_groups": 5,
    "min_seeds": 5,
    "min_captures": 1
  }
}
```

Those criteria are examples. Set them before viewing confirmation outcomes. One capture supports claims scoped to that capture. A lack of independent groups or sufficient seeds is an applicability limit. The receiver in each seed is the one already recorded by the frozen case, and every branch is applied to exactly the same external rows.

For an absolute operating requirement, add `min_absolute_target_f1` or `min_absolute_target_precision` as rates in `[0, 1]` inside `criteria`. Each applies to the mean across all frozen seeds of the federated receiver metric on pooled external rows. The report also gives the minimum and maximum observed seed, including any calibration fallback, and the number of seeds with a defined metric. Choose the floors from the intended operation before scoring the captures. For minimum support on each retained acquisition, declare `min_target_groups_per_capture` and `min_benign_groups_per_capture` as positive integers. Every declared capture must then meet its specified group floors after any overlap exclusions. These optional criteria work with existing Hulk version 5 frozen models and remain absent from the decision when omitted.

## Seal, then confirm

Fill in the [Hulk manifest example](external_preflight.md#hulk-manifest-example)
and save it locally as `external/hulk_manifest.json`. With the benchmark data in place, run
[main.py](../main.py).
The entry locates the five version-7 frozen bundles through the attribution receipt,
validates the manifest, seals the declared plan and then runs confirmation.

Seal reads the manifest, model, source code and development-screen receipts. Hold-out capture files are first read during confirmation. Capture sources supplied by the custodian are verified against their declared hashes. Keep the seal in the dated experiment record and retain custody records separately. The local seal has no trusted timestamp or third-party attestation.

All fixed-alarm bundles bind the receiver's raw TabM weights, TRAIN preprocessing, class order and benign CAL threshold to the subtype heads and a declared attack target. Keep the accompanying TabM checkpoint, overlap index and JSON receipts at the exported paths. Seal checks their hashes before accepting the plan. The frozen bundle version describes its inference protocol.

| Bundle version | Inference protocol |
|---|---|
| 1 (V8) | Each receiver head applies its own frozen CAL threshold. |
| 2 (V9) | The TabM mask is shared, and each head can relabel an existing alarm when the declared target ranks first among the attack classes. |
| 3 (V10) | Each head applies its frozen CAL-fit margin rule, including its abstention state, to the shared TabM alarms. |
| 4 (V11 operating-point comparison) | The guarded pipeline also records the same receiver TabM evaluated at its frozen 1% CAL reference threshold. |
| 5 (V11 CAL operating selection) | The opt-in CAL candidate freezes the selected threshold, the original 1% reference, the CAL group hash and the selection status. New receipts also freeze the reference subtype guard for matched F1 comparison. |
| 6 (V11 class-conditional operating selection) | The candidate freezes one threshold per local TabM attack route for every client, while retaining the original V11 scalar threshold and subtype guard for the reference. |
| 7 (V12 donor attribution) | The scalar CAL-selected detector freezes the single-donor and pooled-donor heads alongside the matched local and full federated heads. |

Confirmation computes the TabM probabilities once per receiver and capture. The fixed-alarm methods share the candidate alarm mask and binary probability scores. The V11 reference applies a second already frozen threshold to the same probabilities. Capture labels enter the evaluation metrics and never enter fitting or threshold selection. Record the inference runtime alongside the seal.

All bundles use the version 1 manifest layout shown above. For fixed-alarm bundles, the criteria may additionally specify `min_target_f1_over_base_pp` (default 1.0) and `max_macro_f1_loss_pp` (default 1.0). The gate comparison requires a positive target F1 paired interval against both the matched local head and original TabM, and a macro-F1 lower bound relative to original TabM within the declared loss limit. The absolute FAR requirement applies to the shared mask. Donor comparisons are reported when that frozen branch is supplied. Each seal binds a single bundle schema and matched method protocol. Use the original source checkout for receipts whose inference source hashes differ. If only the analysis code has changed, the original checkout can confirm the existing seal, or a new seal can be created before accessing the confirmation data.

For a version 4 bundle, each receipt additionally freezes `reference_threshold` and `max_operating_fnr_increase`. The latter is a rate and must agree across seeds. An allowance of `0.01` permits an increase of one percentage point. Confirmation requires the upper endpoint of the 95% seed-paired FNR difference interval to be at most that allowance. The comparison uses pooled held-out rows from the declared receiver and the 1% CAL operating point. The reference's absolute FAR remains an observed outcome. This external workflow measures the receiver; evaluation across the full client population is reported by the development experiment.

Version 5 adds `operating_selection` to the fixed gate. It records the candidate FAR grid, the FAR/FNR/coverage constraints, the selected and reference thresholds, the threshold-fit/operating-select CAL group hash and a selection digest. New receipts keep the original V11 reference threshold from the parent CAL holdout and bind its separately fitted subtype guard. The sealed comparison reports full paired intervals and requires nonnegative mean differences for both target F1 and macro-F1 on matched external receiver rows. The candidate threshold is determined before external evaluation. A client below the declared threshold-fit benign support floor yields `no_feasible_operating_point` and no candidate frozen bundle for that seed. Review the full development decision across seeds before starting external confirmation.

Version 6 also binds `class_thresholds` and `class_routing`. The receiver uses the maximum-scoring attack class from its frozen local TabM to look up the threshold for each row, with class-order tie breaking. The common alarm mask is applied before any local, federated or donor subtype decision. The reference still uses its scalar threshold and its reference subtype guard. The metrics report the receiver's ordered class thresholds alongside the scalar reference. Original version 5 Hulk bundles use the existing scalar inference protocol.

Version 7 records the TRAIN-selected single donor identifier for each seed. All four heads use the same receiver alarm mask and their frozen CAL subtype guards. `paired_intervals.csv` compares the donor pool with the single donor (`donor_pool_vs_single`) and the full federated branch with the single donor (`federated_vs_single`). The existing `receiver_participation` contrast compares the full branch with the donor pool. Complete the attribution freeze before accessing any acquisition used to confirm attribution.

Confirmation emits per-capture and pooled metrics, class support by row and by group, confusion counts, seed-paired differences and a criterion-by-criterion decision in `complete.json`. `capture_metric_summaries.csv` gives the seed mean, range and interval for each capture and each mode. `capture_intervals.csv` weights the available capture-level seed means equally for absolute metrics and paired differences. It gives a descriptive 95% t interval when at least two capture units provide that metric. These intervals describe variation among the declared acquisitions and depend on their independence and representativeness. A single acquisition has no capture-level interval. Fixed-alarm bundles also emit `fixed_alarm_transitions.csv` with class support, baseline alarm recall ceilings and changed predictions. Version 4 reports `base_reference_1pct` in `metrics.csv` and the `operating_fnr_vs_reference` contrast in the paired tables.

The optional, predeclared `min_capture_target_f1_ci95_lower_pp` requires the
lower bound for the federated minus local subtype target-F1 difference to
exceed the specified percentage-point value. It requires `min_captures` of at
least two. The decision uses equal-weight per-capture means from acquisitions
with both target and benign groups and a paired F1 value for every frozen
seed. The receipt records this interval, its support, and aggregated counts
from the overlap screen. A manifest without this criterion uses the existing
pooled, seed-paired acceptance criteria.

`reject` stops when the canonical development fingerprint screen flags an external row. The optional `exclude_overlapping_groups` policy removes every row from any flagged group and records the exclusions. The screen operates on canonical, ordered pre-transform float32 features. It can conservatively flag coincidentally identical legitimate patterns (and in principle hash collisions). It does not infer near-duplicates or session identity from features. Acquisition custody and honest group IDs remain essential.

Development screens, model weights and external captures contain local experiment material. Retain them locally along with the receipts. Aggregate outputs can be reviewed separately for publication.
