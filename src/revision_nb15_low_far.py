"""Train and evaluate a matched low-FAR detector comparison from local NB15 data."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import torch

from .privacy_run import PrivateClient
from .revision_fresh_source import SEEDS, _common_switches
from .revision_nb15_low_far_evaluation import assess_results, evaluate_seed
from .revision_nb15_low_far_fit import EPOCHS, TAIL_FRACTION, fit_low_far
from . import revision_next as native
from .revision_report import write_csv
from .revision_v12 import _bindings, _hash, _write


DATASET = "CIC-UNSW-NB15"
TARGET = "Exploits"
SYNTHETIC_TARGET = "ATTACK_1"
OUTPUT_CONTAINER = "fedcaid_nb15_low_far"
EVAL_ROLES = ("cal", "val")
ARMS = {"frozen_head": "frozen", "full_finetune": "full"}
SOURCE_MODULES = (
    "revision_fresh_source.py",
    "revision_nb15_low_far.py",
    "revision_nb15_low_far_evaluation.py",
    "revision_nb15_low_far_fit.py",
)


def _configuration(project, output, device, smoke):
    """Start with the same controlled-absence design as the main NB15 evidence."""
    parser = native._parser()
    cli = parser.parse_args([
        *_common_switches(project / "dataset", device),
        "--dataset", DATASET,
        "--operating-policy", "fixed",
        "--detector-objective", "prior_adjusted",
        "--out", str(output),
        *(("--smoke",) if smoke else ()),
    ])
    native._normalize(parser, cli)
    return cli


def _source_snapshot(project):
    """Bind original and candidate source without reading old result directories."""
    source = Path(__file__).resolve().parent
    return {
        **native._source_receipt(),
        **_bindings(source, SOURCE_MODULES),
        "main.py": _hash(project / "main.py"),
    }


def _allocate_output(project, container=OUTPUT_CONTAINER, *, output=None):
    """Keep every experiment invocation separate from all previous outputs."""
    if output is not None:
        target = Path(output).absolute()
        if target.resolve() != target:
            raise ValueError("The NB15 stage output cannot be redirected")
        target.mkdir(exist_ok=False)
        return target
    root = project / "outputs" / container
    if root.resolve() != root:
        raise ValueError("The NB15 output container must stay inside the project")
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("run_%Y%m%d_%H%M%S_")
    return Path(mkdtemp(prefix=prefix, dir=root))


def _prepare_seed(data, cli, seed):
    """Reuse grouped allocation, controlled absence and reserved CAL holdouts."""
    name = SYNTHETIC_TARGET if cli.smoke else TARGET
    target = data.names.index(name)
    identity = dict(zip(native.CASE_ID, (DATASET, "train_absence", 0.3, 3, name, seed), strict=True))
    parts, roster, policies, audit = native._design(
        native._scope_data(data), data.names, identity, target, None, cli,
    )
    policy = policies["reserved"]
    if policy["status"] != "eligible":
        raise ValueError(f"Reserved CAL unavailable for seed {seed}: {policy.get('reason')}")
    return parts, roster, policy, audit, native._args(identity, cli)


def _client_rows(roster, policy, client):
    """Use TRAIN as routed and only the original group-disjoint CAL holdout."""
    original, routed = roster[client], policy["roster"][client]
    if not np.array_equal(original["train"], routed["train"]):
        raise ValueError("CAL reservation changed the client TRAIN allocation")
    _, holdout = policy["splits"][client]
    return {"train": original["train"], "cal": routed["cal"][holdout], "val": original["val"]}


def _joint_probabilities(reference, attack_probability):
    """Keep the reference attack subtype ordering while replacing the alarm score."""
    dtype = np.float64
    scores = np.asarray(attack_probability, dtype=dtype)
    if scores.shape != (len(reference),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The candidate must return one finite attack probability per row")
    attack = np.asarray(reference[:, 1:], dtype=dtype)
    total = attack.sum(axis=1, keepdims=True)
    conditional = np.full_like(attack, 1 / attack.shape[1])
    np.divide(attack, total, out=conditional, where=total > 0)
    return np.column_stack((1 - scores, scores[:, None] * conditional))


def _candidate(fit, node):
    """Score CAL and VAL with a trained branch, retaining the reference subtype rule."""
    return tuple(
        _joint_probabilities(node.probabilities[role], fit.score(node.parts[role][0]))
        for role in EVAL_ROLES
    )


def _fit_reference_client(data, parts, rows, args, client, scratch):
    """Fit the original local detector and score only the supplied evaluation roles."""
    local = {role: parts[role] for role in rows}
    client_id = f"client_{client}"
    print(f"FedCAID | NB15 seed={args.seed} {client_id} | 训练参考 TabM", flush=True)
    return PrivateClient(client_id, local, rows, data.names, data.features, args, scratch / client_id)


def _train_client(data, parts, rows, args, client, scratch):
    """Fit a fresh local reference and both candidates on its same TRAIN rows."""
    node = _fit_reference_client(data, parts, rows, args, client, scratch)
    client_id = node.client_id
    scores = {"reference": tuple(node.probabilities[role] for role in EVAL_ROLES)}
    receipt = {"client": client, "reference": node.performance, "candidates": {}}
    for arm, mode in ARMS.items():
        print(f"FedCAID | NB15 seed={args.seed} {client_id} | {arm}", flush=True)
        fit = fit_low_far(
            node.model, *node.parts["train"], mode=mode, seed=args.seed + client,
            device=args.device, batch_size=args.batch_size,
        )
        scores[arm] = _candidate(fit, node)
        receipt["candidates"][arm] = fit.receipt
        del fit
    return tuple(node.parts[role][1] for role in EVAL_ROLES), scores, receipt


def _run_seed(data, cli, seed):
    """Keep only aggregated evidence after the matched scoring of all clients."""
    parts, roster, policy, audit, args = _prepare_seed(data, cli, seed)
    labels, scores, receipts = [], defaultdict(list), []
    with TemporaryDirectory(prefix="fedcaid_nb15_low_far_") as temporary:
        for client in range(len(roster)):
            rows = _client_rows(roster, policy, client)
            truth, probabilities, receipt = _train_client(data, parts, rows, args, client, Path(temporary))
            labels.append(truth)
            receipts.append(receipt)
            for arm, values in probabilities.items():
                scores[arm].append(values)
    tables = evaluate_seed(seed, data.names, audit["receiver"], labels, dict(scores))
    return tables, audit, receipts


def _collect_seeds(output, project, cli, seeds, data, source):
    """Write per-seed outcomes before aggregating paired metrics."""
    tables = defaultdict(list)
    for seed in seeds:
        if _source_snapshot(project) != source:
            raise ValueError("NB15 source changed during the run")
        current, audit, training = _run_seed(data, cli, seed)
        _write(output / f"seed_{seed}.json", {"seed": seed, "audit": audit, "training": training, "tables": current})
        for name, rows in current.items():
            tables[name].extend(rows)
    if _source_snapshot(project) != source:
        raise ValueError("NB15 source changed during the run")
    return tables


def _write_plan(output, project, cli, seeds, source):
    """Declare the controlled development scope and independent-run provenance."""
    _write(output / "plan.json", {
        "dataset": DATASET, "protocol": "grouped", "scenario": "train_absence",
        "seeds": list(seeds), "device": cli.device, "data_root": str(project / "dataset"),
        "target": SYNTHETIC_TARGET if cli.smoke else TARGET,
        "clients": 3, "alpha": 0.3, "synthetic_only": cli.smoke,
        "source_sha256": source, "previous_run_inputs_allowed": False,
        "test_evaluated": False, "selection_on_val": False,
        "reference_epochs": cli.detector_epochs, "candidate_update_epochs": EPOCHS,
        "candidate_train_benign_tail_fraction": TAIL_FRACTION,
        "arms": ["reference", *ARMS], "subtype_rule": "common freshly trained reference subtype",
    })


def _publish(output, tables, seeds):
    """Publish working points, class metrics and a sealed development decision."""
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    assessment = assess_results(tables["working_points"], seeds)
    _write(output / "assessment.json", assessment)
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "complete.json")
    _write(output / "complete.json", {"status": "completed", "artifacts": _bindings(output, names)})
    print(f"FedCAID | NB15 专项结果（{len(seeds)} 个种子） | {output / 'assessment.json'}", flush=True)


def run_nb15_low_far(project, *, output=None, device="auto", smoke=False):
    """Run a fresh NB15-only comparison with no scores or checkpoints from earlier runs."""
    project = Path(project).resolve(strict=True)
    output = _allocate_output(project, output=output)
    cli = _configuration(project, output, device, smoke)
    seeds, source = SEEDS[:1] if smoke else SEEDS, _source_snapshot(project)
    _write_plan(output, project, cli, seeds, source)
    print(f"FedCAID | NB15 低误报专项 | {output}", flush=True)
    if smoke:
        torch.set_num_threads(1)
    data = native._synthetic(DATASET, "grouped") if smoke else native.load_revision_data(
        DATASET, cli.data_root, split_protocol="grouped",
    )
    _write(output / "data_audit.json", data.audit)
    tables = _collect_seeds(output, project, cli, seeds, data, source)
    _publish(output, tables, seeds)
    return output
