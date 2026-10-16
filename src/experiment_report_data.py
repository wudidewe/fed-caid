"""Validate and summarize one completed MPC experiment using JSON artifacts only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ReportDataError(ValueError):
    """An input cannot safely be presented under the report's stated scope."""


SCOPE = {
    "visibility": "private",
    "data_scope": "private researcher diagnostics, outside the public MPC transcript",
    "evaluation_role": "VAL development",
    "test_opened": False,
    "independent_test_evaluated": False,
    "statistical_unit": "one completed experiment",
    "comparison_warning": "No independent TEST or unseen-deployment claim is made.",
    "support_warning": "Support is the recorded VAL sample count for this experiment.",
    "input_content_verified": False,
    "input_verification_scope": "Recorded feature/class contracts and partition identity hashes are validated. Original data contents are not re-read or independently verified by this JSON-only report.",
}
METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "weighted_recall",
    "macro_f1",
    "weighted_f1",
    "far",
    "fnr",
)
OPTIONAL_METRICS = (
    "nll",
    "binary_brier",
    "binary_nll",
    "binary_ece_10",
    "binary_auc",
    "argmax_accuracy",
    "argmax_macro_f1",
    "argmax_weighted_f1",
)
_ROLES = {"train", "cal", "val"}


def _fail(message):
    raise ReportDataError(message)


def _number(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label}: expected a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        _fail(f"{label}: number outside the allowed range")
    return float(value)


def _integer(value, label, minimum=0):
    number = _number(value, label, minimum=minimum)
    if number != int(number):
        _fail(f"{label}: expected an integer")
    return int(number)


def _object(value, label):
    if not isinstance(value, dict):
        _fail(f"{label}: expected a JSON object")
    return value


def _names(value, label, *, path_safe=False):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        _fail(f"{label}: expected unique nonempty names")
    if path_safe and any(item in {".", ".."} or "/" in item or "\\" in item for item in value):
        _fail(f"{label}: path separators are not allowed in dataset names")
    return list(value)


def _same(actual, expected, label):
    if actual != expected:
        _fail(f"{label}: inconsistent input")


def _close(actual, expected, label):
    if expected is None:
        if actual is not None:
            _fail(f"{label}: undefined rate must be null")
    elif actual is None or not math.isclose(
        _number(actual, label), expected, rel_tol=1e-9, abs_tol=1e-10
    ):
        _fail(f"{label}: inconsistent with confusion matrix or selected result")


def _observation(value):
    """Store one observation; ``mean`` is a compatibility key, not an average."""
    return {"mean": value, "sample_sd": None, "n": 0 if value is None else 1, "values": [value]}


class _Reader:
    def __init__(self, root):
        self.root = root
        self.provenance = []

    def read(self, relative):
        path = self.root / relative
        if path.suffix.lower() != ".json" or not path.resolve().is_relative_to(self.root):
            _fail("Report input must be a JSON artifact inside its experiment directory")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ReportDataError(f"Missing or unreadable required artifact: {relative}") from error

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    _fail(f"{relative}: duplicate JSON key {key}")
                result[key] = value
            return result

        def constant(value):
            _fail(f"{relative}: nonfinite JSON value {value}")

        try:
            result = json.loads(
                payload.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ReportDataError(f"Malformed JSON artifact: {relative}") from error

        # Reject overflow values such as JSON 1e999.
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                _fail(f"{relative}: nonfinite JSON number")
            if isinstance(value, dict):
                for item in value.values():
                    finite(item)
            elif isinstance(value, list):
                for item in value:
                    finite(item)

        finite(result)
        self.provenance.append(
            {
                "run_id": self.root.name,
                "file": Path(relative).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        return result

    def exists(self, relative):
        return (self.root / relative).is_file()


def _metrics(raw, label):
    raw = _object(raw, label)
    names = _names(raw.get("class_names"), label + ":class_names")
    if len(names) < 2 or names[0] != "BENIGN":
        _fail(f"{label}: BENIGN must be class zero and at least two classes are required")
    _same(
        raw.get("decision_rule"),
        "total_attack_probability_then_attack_subtype",
        label + ":decision_rule",
    )
    matrix = raw.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(names)
        or any(not isinstance(row, list) or len(row) != len(names) for row in matrix)
    ):
        _fail(f"{label}: confusion matrix must match class order and dimensions")
    matrix = [[_integer(value, label + ":confusion_matrix") for value in row] for row in matrix]
    rows = _integer(raw.get("rows"), label + ":rows", 1)
    _same(sum(map(sum, matrix)), rows, label + ":confusion total")
    per = _object(raw.get("per_class"), label + ":per_class")
    _same(set(per), set(names), label + ":per_class names")
    support, per_class = {}, {}
    for i, name in enumerate(names):
        entry = _object(per[name], label + ":" + name)
        count = _integer(entry.get("support"), label + ":support")
        _same(count, sum(matrix[i]), label + ":support " + name)
        column = sum(row[i] for row in matrix)
        precision = matrix[i][i] / column if column else 0.0
        recall = matrix[i][i] / count if count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for key, expected in (("precision", precision), ("recall", recall), ("f1-score", f1)):
            _close(entry.get(key), expected, label + ":" + name + ":" + key)
        support[name] = count
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": count,
        }
    expected = {"accuracy": sum(matrix[i][i] for i in range(len(names))) / rows}
    for output, key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
        expected["macro_" + output] = sum(per_class[name][key] for name in names) / len(names)
        expected["weighted_" + output] = (
            sum(support[name] * per_class[name][key] for name in names) / rows
        )
    benign, attacks = support[names[0]], rows - support[names[0]]
    expected["far"] = sum(matrix[0][1:]) / benign if benign else None
    expected["fnr"] = sum(row[0] for row in matrix[1:]) / attacks if attacks else None
    result = {}
    for key in METRIC_KEYS:
        if key not in raw and key in {"macro_recall", "weighted_recall"}:
            result[key] = expected[key]
        else:
            _close(raw.get(key), expected[key], label + ":" + key)
            result[key] = None if raw.get(key) is None else float(raw[key])
    for key in OPTIONAL_METRICS:
        if key in raw:
            result[key] = (
                None if raw[key] is None else _number(raw[key], label + ":" + key, minimum=0)
            )
    return {
        "metrics": result,
        "class_names": names,
        "support": support,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "rows": rows,
    }


def _partition(reader, dataset, clients, names):
    signatures, feature_names = {}, None
    for client in clients:
        relative = (
            f"private_DO_NOT_UPLOAD/{dataset}/{client}/prediction_cache_manifest_private.json"
        )
        manifest = _object(reader.read(relative), relative)
        _same(manifest.get("class_names"), names, relative + ":class_names")
        features = _names(manifest.get("feature_names"), relative + ":feature_names")
        if feature_names is not None:
            _same(features, feature_names, relative + ":feature_names")
        feature_names = features
        pins = _object(
            manifest.get("partition_indices_sha256"), relative + ":partition_indices_sha256"
        )
        _same(set(pins), _ROLES, relative + ":partition roles")
        for value in pins.values():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                _fail(f"{relative}: malformed partition SHA-256")
        signatures[client] = pins
    return {"feature_names": feature_names, "partition_indices_sha256": signatures}


def _aggregation(reader, dataset, names, settings):
    rounds, previous = [], [0.0] * (len(names) - 1)
    changed = 0
    for index in range(1, settings["rounds"] + 1):
        relative = f"public/{dataset}/r{index}_aggregate.json"
        raw = _object(reader.read(relative), relative)
        _same(raw.get("schema"), "two_server_additive_sum_v1", relative + ":schema")
        _same(raw.get("round_index", raw.get("round")), index, relative + ":round")
        _same(raw.get("dimension"), len(names) - 1, relative + ":dimension")
        _same(raw.get("roster_size"), settings["clients"], relative + ":roster")
        _same(raw.get("fixed_point_scale"), settings["secure_scale"], relative + ":scale")
        _same(raw.get("clip"), settings["clip"], relative + ":clip")
        if (
            raw.get("all_clients_accepted") is not True
            or raw.get("differential_privacy") is not False
        ):
            _fail(f"{relative}: incomplete roster or incompatible privacy declaration")
        adapter = raw.get("adapter")
        if not isinstance(adapter, list) or len(adapter) != len(names) - 1:
            _fail(f"{relative}: missing or invalid final adapter")
        adapter = [_number(value, relative + ":adapter") for value in adapter]
        nonzero = adapter != previous
        if "accepted_update" in raw and raw["accepted_update"] is not nonzero:
            _fail(f"{relative}: update flag contradicts adapter change")
        changed += int(nonzero)
        counters = {
            key: _number(value, relative + ":" + key, minimum=0)
            for key, value in raw.items()
            if key.endswith("_bytes")
        }
        rounds.append(
            {
                "round": index,
                "adapter": adapter,
                "all_clients_accepted": True,
                "nonzero_aggregate_change": nonzero,
                "communication": counters,
            }
        )
        previous = adapter
    return {
        "completed_rounds": len(rounds),
        "nonzero_update_rounds": changed,
        "all_clients_accepted": True,
        "final_adapter": previous,
        "rounds": rounds,
        "interpretation": "A zero aggregate change is not client rejection or a failed DP gate.",
    }


def _history(reader, dataset, clients, version, epochs):
    if version == 5:
        return {
            "available": False,
            "reason": "v5 is continuation without detector retraining; reference history is not followed outside this run.",
            "clients": {},
        }
    values = {}
    for client in clients:
        relative = f"private_DO_NOT_UPLOAD/{dataset}/{client}/training_history_private.json"
        if not reader.exists(relative):
            return {
                "available": False,
                "reason": "A recorded client TRAIN history is missing; no learning curve is inferred.",
                "clients": {},
            }
        raw = reader.read(relative)
        if not isinstance(raw, list) or len(raw) != epochs:
            _fail(f"{relative}: history must contain the configured epochs")
        series = []
        for index, row in enumerate(raw, 1):
            row = _object(row, relative)
            _same(row.get("epoch"), index, relative + ":epoch order")
            series.append(
                {
                    "epoch": index,
                    "training_loss": _number(
                        row.get("training_loss"), relative + ":loss", minimum=0
                    ),
                    "seconds": _number(row.get("seconds"), relative + ":seconds", minimum=0),
                }
            )
        values[client] = series
    return {
        "available": True,
        "reason": "Recorded local TRAIN objective; not a VAL learning curve.",
        "clients": values,
    }


def _timing(reader, dataset, clients, version):
    relative = f"private_DO_NOT_UPLOAD/{dataset}/performance_private.json"
    raw = _object(reader.read(relative), relative)
    values = {
        key: _number(value, relative + ":" + key, minimum=0)
        for key, value in raw.items()
        if key.endswith("_seconds") or key == "total_seconds"
    }
    required = "total_experiment_seconds" if version == 3 else "total_seconds"
    if required not in values:
        _fail(f"{relative}: missing total recorded stage timer")
    entries = raw.get("clients")
    if not isinstance(entries, list) or len(entries) != len(clients):
        _fail(f"{relative}: client timings do not match the roster")
    output_clients = []
    for index, client in enumerate(clients):
        row = _object(entries[index], relative + ":client")
        _same(row.get("client", client), client, relative + ":client order")
        timers = {
            key: _number(value, relative + ":" + key, minimum=0)
            for key, value in row.items()
            if key.endswith("_seconds")
        }
        inference_rows = None
        if "inference_rows" in row:
            inference_rows = _integer(row["inference_rows"], relative + ":inference_rows", 1)
        if "inference_ms_per_sample" in row:
            recorded = _number(
                row["inference_ms_per_sample"], relative + ":inference_ms_per_sample", minimum=0
            )
            if inference_rows is not None and "inference_seconds" in timers:
                expected = 1000 * timers["inference_seconds"] / inference_rows
                if not math.isclose(recorded, expected, rel_tol=1e-9, abs_tol=1e-10):
                    _fail(
                        relative
                        + ":inference_ms_per_sample disagrees with recorded seconds and rows"
                    )
        output_clients.append(
            {"client": client, "values": timers, "inference_rows": inference_rows}
        )
    if version == 3:
        for key in (
            "detector_training_seconds",
            "preprocessing_seconds",
            "adapter_training_seconds",
        ):
            if all(key in row["values"] for row in output_clients):
                values[key + "_sum_clients"] = sum(row["values"][key] for row in output_clients)
    scope = (
        "Recorded v3 dataset workflow stage timers, including training; not an external whole-process stopwatch and not remote network latency."
        if version == 3
        else "Additional v5 continuation/calibration/selection timers only; excludes original detector training and is not a full-training runtime."
    )
    return {"scope": scope, "values": values, "clients": output_clients}


def _thresholds(reader, dataset, clients, version):
    result = []
    for client in clients:
        relative = f"private_DO_NOT_UPLOAD/{dataset}/{client}/calibration_private.json"
        raw = _object(reader.read(relative), relative)
        value = _number(raw.get("threshold"), relative + ":threshold", minimum=0, maximum=1)
        if raw.get("formal_conformal_guarantee") is not False:
            _fail(f"{relative}: incompatible calibration guarantee")
        if version == 5:
            evidence = f"private_DO_NOT_UPLOAD/{dataset}/{client}/binary_preservation_private.json"
            preserved = _object(reader.read(evidence), evidence)
            if preserved.get("passed") is not True or preserved.get("threshold_exact") is not True:
                _fail(f"{evidence}: selected threshold preservation is unverified")
        result.append(
            {
                "client": client,
                "threshold": value,
                "target_far": _number(
                    raw.get("target_far"), relative + ":target_far", minimum=0, maximum=1
                ),
                "benign_holdout_rows": _integer(
                    raw.get("benign_holdout_rows"), relative + ":benign_holdout_rows"
                ),
                "calibration_far": (
                    None
                    if raw.get("calibration_far") is None
                    else _number(
                        raw["calibration_far"], relative + ":calibration_far", minimum=0, maximum=1
                    )
                ),
                "scope": (
                    "Original binary threshold preserved for selected v5 models."
                    if version == 5
                    else "Client benign CAL holdout empirical threshold; not a formal guarantee."
                ),
            }
        )
    return result


def _load_one(path):
    if not isinstance(path, (str, Path)):
        _fail("Each experiment path must be a string or Path")
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        _fail("Each input must be an existing completed experiment directory")
    reader = _Reader(root)
    completion = _object(reader.read("public/completion.json"), "completion")
    protocol = _object(reader.read("public/protocol.json"), "protocol")
    if completion.get("complete") is not True:
        _fail("Experiment completion must explicitly be true")
    version = completion.get("framework_version")
    if type(version) is not int or version not in {3, 5}:
        _fail("Only completed MPC framework v3 or v5 experiments are supported")
    _same(protocol.get("framework_version"), version, "framework version")
    if (
        completion.get("mechanism") != "mpc"
        or protocol.get("mechanism") != "two_server_additive_secure_sum"
    ):
        _fail("Only MPC experiments are supported; DP reports need a separate scope")
    for document in (completion, protocol):
        if (
            document.get("differential_privacy") is not False
            or document.get("epsilon") is not None
            or document.get("delta") is not None
        ):
            _fail("MPC input must explicitly declare no differential privacy")
        if document.get("test_opened") is not False:
            _fail("Input must explicitly record that TEST remains unopened")
    synthetic = completion.get("synthetic")
    if type(synthetic) is not bool:
        _fail("Completion must explicitly identify real or synthetic input")
    datasets = _names(protocol.get("datasets"), "protocol datasets", path_safe=True)
    _same(
        set(_names(completion.get("datasets"), "completion datasets", path_safe=True)),
        set(datasets),
        "dataset coverage",
    )
    settings = dict(_object(protocol.get("settings"), "settings"))
    _same(settings.get("smoke"), synthetic, "synthetic setting")
    _same(settings.get("mechanism"), "mpc", "settings mechanism")
    training_seed = _integer(settings.get("training_seed", settings.get("seed")), "training seed")
    partition_seed = _integer(
        settings.get("partition_seed", settings.get("seed")), "partition seed"
    )
    settings["training_seed"], settings["partition_seed"], settings["seed"] = (
        training_seed,
        partition_seed,
        partition_seed,
    )
    for key, minimum in (("clients", 3), ("rounds", 1), ("base_epochs", 1), ("secure_scale", 1)):
        settings[key] = _integer(settings.get(key), "settings:" + key, minimum)
    settings["clip"] = _number(settings.get("clip"), "settings:clip", minimum=0)
    if settings["clip"] <= 0:
        _fail("Clipping radius must be positive")
    dataset_parameters = protocol.get("dataset_parameters", {})
    if not isinstance(dataset_parameters, dict):
        _fail("dataset_parameters must be an object")
    if dataset_parameters and set(dataset_parameters) != set(datasets):
        _fail("dataset_parameters coverage differs")
    clients = [f"client_{index}" for index in range(settings["clients"])]
    result = {
        "run_id": root.name,
        "training_seed": training_seed,
        "partition_seed": partition_seed,
        "framework_version": version,
        "synthetic": synthetic,
        "settings": settings,
        "datasets": {},
    }
    source_manifest = "public/execution_source_hashes.json"
    recorded_sources = {}
    if reader.exists(source_manifest):
        recorded_sources = _object(reader.read(source_manifest), source_manifest)
        for name, digest in recorded_sources.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name:
                _fail("Execution source hashes must use relative filenames")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                _fail("Malformed recorded execution source SHA-256")
    result["source_hashes"] = {
        "recorded": recorded_sources,
        "content_rechecked": False,
        "scope": "Original recorded source hashes; this report reads JSON only and does not reopen source or binary artifacts.",
    }
    for dataset in datasets:
        override = _object(dataset_parameters.get(dataset, {}), "dataset parameters:" + dataset)
        current_settings = {**settings, **override}
        for key, minimum in (
            ("clients", 3),
            ("rounds", 1),
            ("base_epochs", 1),
            ("secure_scale", 1),
        ):
            current_settings[key] = _integer(
                current_settings.get(key), dataset + ":" + key, minimum
            )
        current_settings["clip"] = _number(
            current_settings.get("clip"), dataset + ":clip", minimum=0
        )
        if current_settings["clip"] <= 0:
            _fail("Dataset clipping radius must be positive")
        clients = [f"client_{index}" for index in range(current_settings["clients"])]
        private = f"private_DO_NOT_UPLOAD/{dataset}"
        selected = f"private_DO_NOT_UPLOAD/selected_models/{dataset}"
        metric_path = (selected if version == 5 else private) + "/pooled_validation_private.json"
        data = _metrics(reader.read(metric_path), metric_path)
        selection = {"arm": "v3", "scope": "Default v3 pooled development metrics."}
        if version == 5:
            relative = private + "/release_guard_private.json"
            guard = _object(reader.read(relative), relative)
            arm = guard.get("selected")
            if arm not in {"candidate", "reference"} or guard.get("passed") is not (
                arm == "candidate"
            ):
                _fail(f"{relative}: selected arm contradicts guard result")
            for key, comparison in _object(
                guard.get("comparisons"), relative + ":comparisons"
            ).items():
                if key in data["metrics"] and isinstance(comparison, dict) and arm in comparison:
                    _close(
                        data["metrics"][key], comparison[arm], relative + ":selected metric " + key
                    )
            selection = {
                "arm": arm,
                "guard_passed": guard["passed"],
                "scope": "Selected final models; VAL was used for development release selection, not independent evaluation.",
            }
        data["selection"] = selection
        data["data_signature"] = _partition(reader, dataset, clients, data["class_names"])
        data["thresholds"] = _thresholds(reader, dataset, clients, version)
        data["aggregation"] = _aggregation(reader, dataset, data["class_names"], current_settings)
        data["timing"] = _timing(reader, dataset, clients, version)
        data["history"] = _history(
            reader, dataset, clients, version, current_settings["base_epochs"]
        )
        result["datasets"][dataset] = data
    result["dataset_parameters"] = dataset_parameters
    return result, datasets, reader.provenance


def load_runs(paths):
    """Validate exactly one completed run using required JSON artifacts only."""
    if isinstance(paths, (str, bytes, Path)):
        _fail("Pass a list of experiment directories, not one path string")
    try:
        paths = list(paths)
    except TypeError as error:
        raise ReportDataError("paths must be an iterable of directories") from error
    if len(paths) != 1:
        _fail("Exactly one completed experiment is required; multiple runs are not supported")
    run, datasets, provenance = _load_one(paths[0])
    configuration = {key: value for key, value in run["settings"].items() if key != "training_seed"}
    if run.get("dataset_parameters"):
        configuration["dataset_parameters"] = run["dataset_parameters"]
    summary = {}
    for dataset in datasets:
        first = run["datasets"][dataset]
        per_class = {}
        for name in first["class_names"]:
            per_class[name] = {
                output: _observation(first["per_class"][name][key])
                for output, key in (
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("f1", "f1-score"),
                )
            }
            per_class[name]["support"] = first["support"][name]
        summary[dataset] = {
            "class_names": first["class_names"],
            "support": first["support"],
            "rows": first["rows"],
            "metrics": {
                key: _observation(value) for key, value in sorted(first["metrics"].items())
            },
            "per_class": per_class,
            "confusion": {"matrix": first["confusion_matrix"]},
            "thresholds": [{"training_seed": run["training_seed"], "clients": first["thresholds"]}],
            "aggregation": [{"training_seed": run["training_seed"], **first["aggregation"]}],
            "timing": {
                "scope": first["timing"]["scope"],
                "metrics": {
                    key: _observation(value)
                    for key, value in sorted(first["timing"]["values"].items())
                },
            },
        }
    scope = dict(SCOPE)
    scope["evidence_kind"] = (
        "Synthetic functional check, not an intrusion-detection benchmark."
        if run["synthetic"]
        else "Existing development observations, not independent TEST evidence."
    )
    return {
        "schema_version": 1,
        "scope": scope,
        "framework_version": run["framework_version"],
        "synthetic": run["synthetic"],
        "datasets": datasets,
        "seeds": [run["training_seed"]],
        "run_count": 1,
        "configuration": configuration,
        "runs": [run],
        "summary": summary,
        "provenance": provenance,
        "privacy_scope": "Local two-collector MPC; noncollusion and trusted host/coordinator assumed; no DP, remote deployment, or attack-resistance proof.",
    }
