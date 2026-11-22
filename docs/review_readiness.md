# Validation and reproduction

The [FedCAID workflow](../README.md) links each saved result to its method
and evaluation population. The original class offset has K−1 coordinates. The
guarded shared linear subtype head has K(d+1) trainable parameters. FedAvg repair
uses a separate full-weight donor. Use the corresponding communication rows for each.

| Area | Executable evidence | Scope |
| --- | --- | --- |
| Federated contribution | Local, federated and donor controls with paired estimates | Development comparisons on matched evaluation rows |
| Data separation | Five seeds, TRAIN and CAL target selection, exact input and role audits | Grouped development VAL for the main matrix. Retrospective TEST views for selected studies. |
| Minority classes | Per class support, precision, recall and F1 | Interpret class metrics with support, FAR and FNR for the same view. |
| Aggregation protocol | Local subprocess sharing, TCP protocol validation and failure tests | Trusted local host and noncolluding collectors |
| Software comparison | Primary source feature comparison | Functional comparison |
| Scale and cost | Fixed total and fixed client budgets, logical payload and framed application bytes | Separate detector, adapter and transport measurements |
| Reproducibility | CPU demo and integration checks, wheel installation checker, CI and run manifests | The run receipt identifies the source, settings and available inputs. |

The Windows and Linux CI jobs use `tools/check_installed_wheel.py --resolve-dependencies --cpu-torch`
to install the wheel and its declared dependencies into a clean temporary venv,
run `pip check`, import the installed `src` package, and exercise synthetic
inference outside the checkout.
The receipt records both `dependency_resolution_tested` and `complete`. The
job selects PyTorch from its official CPU wheel index.

The [reference source commit](https://github.com/wudidewe/fed-caid/commit/6f72abd255d4782302fd27ac35f8061f529ec141)
matches the Python source recorded by the 17 November 2026 run. See
[review evidence](review_evidence.md) for the report tables and evaluation
populations. Independent acquisition and LAN measurements use the optional
declarations in [external preflight](external_preflight.md) and
[protocol validation](protocol_validation.md).

Threat model: client code, host and coordinator are trusted; clients clip
honestly; collectors are honest-but-curious and do not collude. Each collector
sees a random share, roster/dimensions and protocol metadata. The coordinator
learns the complete-roster aggregate. The trusted local host can access local
data. Collusion, malicious updates/coordinator, host compromise, DP, transport
authentication/confidentiality and dropout recovery are outside current scope.
The [unit suite used for the manuscript](https://github.com/wudidewe/fed-caid/tree/29fc027963374bb05b9d3f783ad942e59bd8323d/tests)
checks single share uniformity and failure behavior for the reported revision.

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
