# Software positioning and evidence scope

Checked primary publisher pages, author repositories and release tags on
2026-11-15. This is a functional comparison, not a benchmark ranking. A source
snapshot is not necessarily the version evaluated in its paper. Unknown release
bindings and unaudited features are explicitly marked unverified.

| Software | Verified positioning | Appropriate comparison with FedCAID | Primary source |
| --- | --- | --- | --- |
| CIDS-Sim | Federated collaborative IDS simulation, FedAvg, non-IID/heterogeneous experiments and resource logs | Partitioning, simulation scope, metrics and communication records | [Paper, code metadata v1.0](https://doi.org/10.1016/j.softx.2026.102511), [source snapshot](https://github.com/aulwardana/CIDS-Sim/tree/76676c885beea0c0e69bd815976d604ccdfee36f) |
| SHAVA | SHAP/MASV feature selection and weighted SMOTE for IDS, including desktop/edge evaluation | Feature processing, minority attacks, inference environment | [Paper](https://www.sciencedirect.com/science/article/pii/S2352711026001846) |
| SB-Net | Feature selection and two-stage cascade detection software | Detection pipeline, task/label definitions and reproduction entry | [Author maintained source snapshot](https://github.com/algof/SB-Net/tree/57d85b0b754a97743deddab896c614b77ff5c16c) |
| Flower | General FL infrastructure; official SecAgg+ simulation/deployment example | Strategy extensibility, transport and security scope | [Release 1.43.0](https://github.com/flwrlabs/flower/releases/tag/framework-1.43.0), [pinned SecAgg+ example](https://github.com/flwrlabs/flower/blob/4f9be0fd56b82c60112d81dc78b7961445381ca7/examples/flower-secure-aggregation/README.md) |
| FedML | ML/FL training and deployment infrastructure | Deployment modes, strategies and configuration | [Release 0.8.9](https://github.com/FedML-AI/FedML/releases/tag/v0.8.9), [pinned source](https://github.com/FedML-AI/FedML/tree/f352cfb5a59f63d4f85ede43682962a5f9ea4a6a) |
| FATE | Federated algorithms and secure computation infrastructure including HE/MPC | Trust model, secure computation and deployment scope | [Release 2.2.0](https://github.com/FederatedAI/FATE/releases/tag/v2.2.0), [pinned source](https://github.com/FederatedAI/FATE/tree/d4d8dd8557397c149634709ac319b544375ec0ae) |
| FedCAID | Four dataset-specific pipelines, role/hash audits, matched ablations, versioned reports and local additive-share simulation | Integrated audit/reproduction workflow; optional full-weight repair and experimental conditional residual | This release's code and run manifests |

Version bindings used for this comparison:

| Software | Publication/release identity | Exact source binding and limitation |
| --- | --- | --- |
| CIDS-Sim | SoftwareX 33 (2026), 102511; the paper declares code v1.0 | Repository snapshot `76676c885beea0c0e69bd815976d604ccdfee36f` (commit date 2025-05-05). The paper links the repository but the equivalence of this commit to paper v1.0 is **unverified**. |
| SHAVA | SoftwareX 34 (2026), 102692 | Publisher abstract and methodology positioning checked. Published source release, author repository and exact commit are **unverified**; do not infer them from an unrelated repository with the same name. |
| SB-Net | Maintained repository README identifies SoftwareX DOI `10.1016/j.softx.2026.102672` | Snapshot `57d85b0b754a97743deddab896c614b77ff5c16c` (2026-05-05) in `algof/SB-Net` contains fixes/documentation relative to the [original repository snapshot](https://github.com/IrsyadFikriansyah/SB-Net/tree/6f7f11c2b3c3a47c864748b513ee607e60b4f499). Binding either snapshot to the published experiment/version is **unverified**. |
| Flower | Official release/tag `framework-1.43.0` | `4f9be0fd56b82c60112d81dc78b7961445381ca7`; tag resolved and SecAgg+ example read at this commit. |
| FedML | Official release/tag `v0.8.9` | `f352cfb5a59f63d4f85ede43682962a5f9ea4a6a`; tag resolved and README checked. This chosen stable reference is not a claim that all current platform features exist in that tag. |
| FATE | Official release/tag `v2.2.0` | `d4d8dd8557397c149634709ac319b544375ec0ae`; tag resolved and README checked. |
| FedCAID | Reference experiment run on 17 November 2026 | The [reference source commit](https://github.com/wudidewe/fed-caid/commit/6f72abd255d4782302fd27ac35f8061f529ec141) matches the experiment's Python source hashes. |

The three framework tags were resolved through their official GitHub repositories;
the two IDS source snapshots were obtained from the author repositories on the
check date. Reading a release/README does not constitute an installation or a
reproduction of the other software's performance.

For each final publication table, separately record supported inputs, aggregation
object, threat assumptions, deployment modes, leakage checks, reports, testing,
license and exact version. Dataset-role hashing and all reporting capabilities
of other tools have not been exhaustively audited here; mark these cells
unverified rather than claiming exclusive functionality.

FedCAID does not claim greater accuracy, security or scale than these frameworks.
If empirical framework comparison is required, implement the same FedAvg model,
preprocessing, partitions, initialization, local steps, thresholds and hardware
in one framework. Do not reuse scores from incompatible papers as baselines.
No external-framework performance experiment has been completed by this change.
