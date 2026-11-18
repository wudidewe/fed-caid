"""Create review tables and a payload figure from a completed real run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import file_digest, sha256
from itertools import product
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from . import revision_report
from ._iov_layout import project_root
from .experiment_config import DATASETS
from .revision_report import METRICS, PAIRS, summarize, write_csv

LOW_SUPPORT = 30
MAIN_MODES = ("base", "local_only", "full", "fedavg", "fedavg_threshold")
ROLES = ("val", "test", "test_unique_unseen")
SCENARIOS = {
    "main",
    "balance",
    "scale_reference",
    "scale_alpha_low",
    "scale_alpha_high",
    "scale_clients_5",
    "scale_clients_10",
}


def _digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _report_source_hashes():
    return {
        "review_evidence.py": _digest(Path(__file__)),
        "revision_report.py": _digest(Path(revision_report.__file__)),
    }


def support_status(n):
    return "absent" if n == 0 else "low_support" if n < LOW_SUPPORT else "at_least_30"


def _load(run):
    hashes = {}

    def read(relative):
        payload = (run / relative).read_bytes()
        hashes[relative] = sha256(payload).hexdigest()
        return json.loads(payload)

    meta, complete = read("run.json"), read("complete.json")
    if meta.get("experiment") != "fedcaid_review" or meta.get("synthetic_only") or meta.get("data_only"):
        raise ValueError("Choose a completed real FedCAID revision experiment")
    seeds = tuple(meta["seeds"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or not all(type(seed) is int for seed in seeds):
        raise ValueError("Review tables require at least five distinct seeds")
    expected = set()
    audits, results = {}, {}
    for row in meta["matrix"]:
        d, p = row["dataset"], row["protocol"]
        if d not in DATASETS or p not in ("frozen", "grouped") or (d, p) in audits:
            raise ValueError(f"Invalid dataset/protocol in the run matrix: {d}, {p}")
        audit = read(f"{d}/{p}/data_audit.json")
        audits[d, p] = audit
        for scenario, clients, alpha in row["scenarios"]:
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            for seed in seeds:
                case = d, p, scenario, seed
                if case in expected:
                    raise ValueError(f"Duplicate experiment case: {case}")
                expected.add(case)
                result = read(f"{d}/{p}/{scenario}/seed_{seed}/result.json")
                if tuple(
                    result[k] for k in ("dataset", "protocol", "scenario", "seed", "clients", "alpha")
                ) != (d, p, scenario, seed, clients, alpha):
                    raise ValueError(f"Result identity mismatch: {case}")
                if scenario == "main":
                    results.setdefault((d, p), []).append(result)
    actual = {path.relative_to(run).parent.as_posix() for path in run.rglob("result.json")}
    declared = {f"{d}/{p}/{s}/seed_{seed}" for d, p, s, seed in expected}
    if (
        actual != declared
        or complete["replicates"] != len(expected)
        or complete["data_audits"] != len(audits)
    ):
        raise ValueError("The result files do not match the completed run matrix")
    if (
        not results
        or set(results) != set(audits)
        or any(len(cases) != len(seeds) for cases in results.values())
    ):
        raise ValueError("Each selected dataset/protocol needs its completed main scenario")
    return seeds, audits, results, hashes


def _support(audit):
    exact = audit["exact_feature_audit"]
    supports = {
        **audit["role_class_support"],
        "test_unique_unseen": exact["test_exact_input_unseen_deduplicated"],
    }
    for role, counts in audit["role_rows"].items():
        if sum(supports[role].values()) != counts:
            raise ValueError(f"Role support does not match total rows: {role}")
    return supports


def population_support(counts):
    """Describe what a fixed evaluation population can actually measure."""
    if "BENIGN" not in counts or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Class support needs BENIGN and nonnegative integer counts")
    rows, benign = sum(counts.values()), counts["BENIGN"]
    observed = sum(n > 0 for n in counts.values())
    return {
        "rows": rows,
        "benign_rows": benign,
        "attack_rows": rows - benign,
        "classes_defined": len(counts),
        "classes_present": observed,
        "attack_classes_present": sum(n > 0 for name, n in counts.items() if name != "BENIGN"),
        "absent_classes": "|".join(name for name, n in counts.items() if n == 0),
        "low_support_classes": "|".join(name for name, n in counts.items() if 0 < n < LOW_SUPPORT),
        "population_status": (
            "empty"
            if not rows
            else (
                "single_class"
                if observed == 1
                else "partial_class_support" if observed < len(counts) else "all_classes_present"
            )
        ),
        "multiclass_metrics_evaluable": observed >= 2,
        "far_evaluable": benign > 0,
        "fnr_evaluable": rows > benign,
    }


def _metric_support(metric, population):
    if metric in ("macro_f1", "balanced_accuracy"):
        status = (
            "evaluable"
            if population["multiclass_metrics_evaluable"]
            else "unavailable_fewer_than_two_classes"
        )
    elif metric == "far":
        status = "evaluable" if population["far_evaluable"] else "unavailable_no_benign_rows"
    elif metric == "fnr":
        status = "evaluable" if population["fnr_evaluable"] else "unavailable_no_attack_rows"
    else:
        status = "evaluable" if population["rows"] else "unavailable_empty_population"
    return {"population_status": population["population_status"], "metric_support_status": status}


def _tables(audits, results):
    metrics, paired, class_support, class_recall, split, payload = [], [], [], [], [], []
    populations = []
    for (dataset, protocol), cases in results.items():
        key = {"dataset": dataset, "protocol": protocol}
        audit = audits[dataset, protocol]
        supports = _support(audit)
        population_by_role = {role: population_support(counts) for role, counts in supports.items()}
        populations.extend(
            {**key, "role": role, **population} for role, population in population_by_role.items()
        )
        exact = audit["exact_feature_audit"]
        overlap = exact["cross_role_exact_overlap"]
        split.append({
            **key,
            **{f"{role}_rows": n for role, n in audit["role_rows"].items()},
            "test_rows_matching_train": overlap["train:test"]["test_rows"],
            "cross_role_pairs_with_exact_matches": sum(
                p["unique_shared_inputs"] > 0 for p in overlap.values()
            ),
            "test_unique_unseen_rows": sum(supports["test_unique_unseen"].values()),
            "test_classes_present": sum(n > 0 for n in supports["test"].values()),
            "test_unique_unseen_classes_present": sum(n > 0 for n in supports["test_unique_unseen"].values()),
            "classes_defined": len(supports["test"]),
            "external_captures": audit["external_captures"],
            **{
                f"{role}_{field}": population_by_role[role][field]
                for role in ("test", "test_unique_unseen")
                for field in ("benign_rows", "attack_rows", "population_status")
            },
        })
        lookup, recall, communication = {}, defaultdict(list), defaultdict(list)
        for case in cases:
            for item in case["communication"]:
                if item["mode"] in {"full", "fedavg"}:
                    communication[item["mode"]].append(item["round_payload_bytes"])
            for item in case["evaluations"]:
                role, mode = item["role"], item["mode"]
                if role not in ROLES:
                    continue
                got = {row["class"]: row["support"] for row in item["per_class"]}
                if got != supports[role] or item["metrics"]["rows"] != sum(got.values()):
                    raise ValueError(
                        f"Unexpected support in {dataset} {protocol} seed {case['seed']} {role} {mode}"
                    )
                index = (case["seed"], role, mode)
                if index in lookup:
                    raise ValueError(f"Duplicate evaluation for {dataset} {protocol}: {index}")
                lookup[index] = item["metrics"]
                if role != "val" and mode == "full":
                    for row in item["per_class"]:
                        recall[role, row["class"]].append(row["recall"])
        for role, counts in supports.items():
            for label, n in counts.items():
                identity = {
                    **key,
                    "role": role,
                    "class": label,
                    "support": n,
                    "support_status": support_status(n),
                }
                class_support.append(identity)
                if role in ("test", "test_unique_unseen"):
                    class_recall.append({**identity, "mode": "full", **summarize(recall[role, label])})
        for role in ROLES:
            by_mode = defaultdict(list)
            for case in cases:
                for seed, candidate_role, mode in lookup:
                    if candidate_role == role and seed == case["seed"]:
                        by_mode[mode].append(lookup[seed, role, mode])
            if set(MAIN_MODES) - by_mode.keys() or any(
                len(values) != len(cases) for values in by_mode.values()
            ):
                raise ValueError(f"Incomplete matched evaluations for {dataset} {protocol} {role}")
            for mode in MAIN_MODES:
                for metric in METRICS:
                    metrics.append({
                        **key,
                        "role": role,
                        "mode": mode,
                        "metric": metric,
                        **_metric_support(metric, population_by_role[role]),
                        **summarize([row[metric] for row in by_mode[mode]]),
                    })
            for contrast, (left, right) in PAIRS.items():
                # PAIRS also contains supplement-only modes. A main report must
                # not manufacture empty lists for modes absent from this run.
                if left not in by_mode or right not in by_mode:
                    continue
                for metric in METRICS:
                    matched = zip(by_mode[left], by_mode[right], strict=True)
                    diffs = [
                        a[metric] - b[metric]
                        for a, b in matched
                        if a[metric] is not None and b[metric] is not None
                    ]
                    paired.append({
                        **key,
                        "role": role,
                        "contrast": contrast,
                        "metric": metric,
                        **_metric_support(metric, population_by_role[role]),
                        "positive_seeds": sum(value > 1e-12 for value in diffs),
                        "negative_seeds": sum(value < -1e-12 for value in diffs),
                        "tied_seeds": sum(abs(value) <= 1e-12 for value in diffs),
                        **summarize(diffs, bounds=(-1, 1)),
                    })
        for mode in ("full", "fedavg"):
            values = communication[mode]
            if len(values) != len(cases) or any(type(v) is not int or v <= 0 for v in values):
                raise ValueError(f"Incomplete numeric communication payloads for {dataset} {protocol} {mode}")
            for role in ("test", "test_unique_unseen"):
                f1 = summarize([lookup[c["seed"], role, mode]["macro_f1"] for c in cases])
                payload.append({
                    **key,
                    "role": role,
                    "mode": mode,
                    "numeric_round_bytes_median": median(values),
                    "numeric_round_bytes_min": min(values),
                    "numeric_round_bytes_max": max(values),
                    **{f"macro_f1_{k}": value for k, value in f1.items()},
                })
    return {
        "main_seed_intervals": metrics,
        "main_paired_intervals": paired,
        "class_support": class_support,
        "class_test_recall": class_recall,
        "split_audit_summary": split,
        "communication_tradeoff": payload,
        "evaluation_population_support": populations,
    }


def _payload_figure(output, payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {
        (r["dataset"], r["mode"]): r for r in payload if r["protocol"] == "grouped" and r["role"] == "test"
    }
    datasets = [dataset for dataset in DATASETS if (dataset, "full") in data]
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for mode, offset, marker, color, label in (
        ("full", -0.14, "o", "#12557D", "FedCAID adapter"),
        ("fedavg", 0.14, "s", "#9C481E", "FedAvg weights"),
    ):
        rows = [data[d, mode] for d in datasets]
        center = [r["numeric_round_bytes_median"] for r in rows]
        err = [
            [v - r["numeric_round_bytes_min"] for v, r in zip(center, rows)],
            [r["numeric_round_bytes_max"] - v for v, r in zip(center, rows)],
        ]
        ax.errorbar(
            center,
            [i + offset for i in range(len(rows))],
            xerr=err,
            marker=marker,
            linestyle="none",
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xscale("log")
    ax.set_xlabel("Logical numeric bytes per round")
    ax.invert_yaxis()
    ax.grid(axis="x", which="major", color="#d0d0d0", linewidth=0.6)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output / "communication_payload.pdf", bbox_inches="tight")
    fig.savefig(output / "communication_payload.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _reuse_verified(output, run, hashes, figures, expected):
    """Accept a previous report only when its manifest and all exports agree."""
    try:
        summary = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
        files = summary["output_sha256"]
        valid = (
            summary["source_run"] == str(run)
            and summary["input_sha256"] == hashes
            and summary.get("report_source_sha256") == _report_source_hashes()
            and summary["figures"] is figures
            and set(files) == expected
            and {path.name for path in output.iterdir()} == expected | {"evidence.json"}
            and all(_digest(output / name) == value for name, value in files.items())
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise FileExistsError(
            f"Existing evidence could not be verified. Choose a new output directory: {output}"
        )
    return output


def generate(run, output, *, figures=True):
    run, output = Path(run).resolve(), Path(output).resolve()
    seeds, audits, results, hashes = _load(run)
    tables = _tables(audits, results)
    expected = {f"{name}.csv" for name in tables}
    if figures and any(protocol == "grouped" for _, protocol in results):
        expected.update({"communication_payload.pdf", "communication_payload.png"})
    if output.exists():
        return _reuse_verified(output, run, hashes, figures, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="evidence_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, rows in tables.items():
            write_csv(stage / f"{name}.csv", rows)
        if figures:
            _payload_figure(stage, tables["communication_tradeoff"])
        summary = {
            "source_run": str(run),
            "seeds": list(seeds),
            "main_replicates": sum(map(len, results.values())),
            "figures": figures,
            "low_support": f"1 to {LOW_SUPPORT - 1} observations in the specific evaluation population",
            "interval": (
                "95% Student t interval for the mean across seeds on a fixed evaluation population,"
                " intersected with the metric domain"
            ),
            "paired_interval": (
                "Difference within each seed, then a Student t interval across those differences"
            ),
            "scope": (
                "Seed variability only; these intervals do not estimate uncertainty across independent"
                " captures"
            ),
            "evaluation_support": (
                "Read evaluation_population_support.csv with metric_support_status; single-class or"
                " absent-attack populations cannot establish general multiclass or attack-recall performance"
            ),
            "metric_units": "Fractions; multiply by 100 for percentages and percentage-point differences",
            "payload_scope": "Logical numeric bytes, excluding serialization and network overhead",
            "input_sha256": hashes,
            "report_source_sha256": _report_source_hashes(),
            "output_sha256": {name: _digest(stage / name) for name in sorted(expected)},
        }
        (stage / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return output


def latest_real_run(root):
    full_profiles = set(product(DATASETS, ("frozen", "grouped")))
    candidates = sorted(
        Path(root).glob("revision*/complete.json"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    for marker in candidates:
        try:
            complete = json.loads(marker.read_text(encoding="utf-8"))
            config = json.loads((marker.parent / "run.json").read_text(encoding="utf-8"))
            matrix = config["matrix"]
            profiles = {(row["dataset"], row["protocol"]) for row in matrix}
            expected = len(config["seeds"]) * sum(len(row["scenarios"]) for row in matrix)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        if (
            config.get("experiment") == "fedcaid_review"
            and not config.get("synthetic_only")
            and not config.get("data_only")
            and len(matrix) == complete.get("data_audits") == len(full_profiles)
            and profiles == full_profiles
            and len(set(config["seeds"])) >= 5
            and expected == complete.get("replicates")
            and expected in {90, 110}
            and all(any(scenario[0] == "main" for scenario in row["scenarios"]) for row in matrix)
        ):
            return marker.parent
    raise ValueError("No completed full revision run found. Select a completed run with --run.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        run = args.run or latest_real_run(project_root(__file__) / "outputs")
        output = generate(run, args.out or run / "review_evidence", figures=not args.no_figures)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"EVIDENCE STOP: {error}\n")
    print(f"EVIDENCE COMPLETE: {output}")
