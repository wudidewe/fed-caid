# Review evidence and remaining external dependencies

The [FedCAID workflow](../README.md) links each saved result to its method
and evaluation population. The original class offset has K−1 coordinates. The
guarded shared linear subtype head has K(d+1) trainable parameters. FedAvg repair
uses a separate full-weight donor. Use the corresponding communication rows for each.

| Review concern | Executable evidence in this revision | What must still be demonstrated |
| --- | --- | --- |
| Federated value | Conditional local/federated matched fits, old bias controls, donor-only cells, paired development decisions | Real-data efficacy; independent frozen evaluation after model development |
| Robustness/leakage | Five-seed real plans, TRAIN/CAL-only target selection, role audits, VAL-only defaults | New independent capture/session data where feasible; reviewed TEST is exploratory |
| Minority classes | Controlled VAL support/precision/recall/F1, per-class existing tables, cache threshold replay | Actual class-specific improvements without unacceptable FAR changes |
| Security | Local two-process sharing plus TCP protocol validation and failure tests | Separate-host deployment; no guarantee against malicious parties or collusion |
| Software comparison | Primary-source functional comparison | Same-configuration external framework timing if claimed |
| Scale/cost | Fixed-total/fixed-local plans; actual framed message bytes; isolated cached adapter profiling | Real hardware/network measurements; adapter-only profiling excludes detector costs |
| Reproducibility | Tests, smoke, wheel installation checker, CI configuration, provenance and resume checks | Public CI success, version/tag, archive DOI/capsule, author and dataset permission confirmation |

Completion of a script cannot set all review requirements to complete. No DOI,
public CI result, author contribution, data redistribution right or external
validation is fabricated by the software. New experimental code is recorded as
unreleased; historical runs retain their original manifests.

The Windows and Linux CI jobs use `tools/check_installed_wheel.py --resolve-dependencies --cpu-torch`
to install the wheel and its declared dependencies into a clean temporary venv,
run `pip check`, import the installed `src` package, and exercise synthetic
inference outside the checkout.
The receipt records both `dependency_resolution_tested` and `complete`. The
job selects PyTorch from its official CPU wheel index.

The paper still needs the original reviewers' full parameter table, software
and code metadata tables, role/support table, CAN confusion matrices, equations
(basis/prior/prox/clipping/encoding), readable figures, architecture/data-flow
diagram, IoV proxy scope, NB15 source clarification, reference verification and
accurate CRediT. Use the actual SoftwareX template at submission time.

Threat model: client code, host and coordinator are trusted; clients clip
honestly; collectors are honest-but-curious and do not collude. Each collector
sees a random share, roster/dimensions and protocol metadata. The coordinator
learns the complete-roster aggregate. The trusted local host can access local
data. Collusion, malicious updates/coordinator, host compromise, DP, transport
authentication/confidentiality and dropout recovery are outside current scope.
Uniform-share algebra and boundary tests support this limited claim, not a
general production security certification.

| Role | Model | What the role sees or controls |
| --- | --- | --- |
| Client | Trusted to follow clipping, quantization and sharing | Its local data and update, the round contract, and the broadcast aggregate model |
| Local host | Trusted | Can access local process state and data in the simulation |
| Coordinator | Trusted | Enforces the complete roster and reconstructs the aggregate from both aggregate shares |
| Each collector | Honest-but-curious and noncolluding | Sees only its own per-client shares plus the contract and delivery metadata |

The mechanism is two-collector additive secret sharing (local simulation).
An incomplete roster causes the round to abort without an accepted aggregate.
Report operator-declared LAN topology separately from independent hardware
verification and from local training performance.
