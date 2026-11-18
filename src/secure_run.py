"""Shared two-server secure-sum experiment; aggregate confidentiality, no DP claim."""

from __future__ import annotations

import csv
import gc
import hashlib
import math
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from . import utils as core
from .aggregation_audit import AggregationAudit
from .privacy_core import ClassScoreAdapter, digest
from .secure_aggregation import SecureAggregationSession
from .validation_core import privacy_scope_validation


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode("utf-8"))
        result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _benchmark_inference(node, raw, device, batch_size):
    """Warm resident model; measure transform through final decision, locally."""
    from .privacy_run import hierarchical_decisions

    model = node.model.to(device)
    correction = ClassScoreAdapter(len(node.names))
    correction.set_vector(node.final_local_vector)

    def infer(values):
        p = core.probabilities(core.log_prediction(model, node.transform(values), device, batch_size))
        p = correction.transform(p)
        calibration = getattr(node, "attack_subtype_calibration", None)
        scores = calibration.scores(p) if calibration is not None and not calibration.identity else None
        from .privacy_run import subtype_joint_probabilities

        return subtype_joint_probabilities(p, scores), hierarchical_decisions(p, node.alarm_threshold, scores)

    def timed(values):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        p, decisions = infer(values)
        if device == "cuda":
            torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - started), p, decisions

    # Exclude warm-up; timing repeats do not retrain or publish new updates.
    infer(raw[: min(batch_size, len(raw))])
    batch_ms = []
    for _ in range(3):
        elapsed, p, decisions = timed(raw)
        batch_ms.append(elapsed)
    difference = float(np.max(np.abs(p - node.final_probabilities)))
    if not np.allclose(p, node.final_probabilities, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Resident end-to-end inference differs from saved cached predictions")
    for _ in range(3):
        infer(raw[:1])
    request_ms = [timed(raw[:1])[0] for _ in range(20)]
    node.model = model.cpu()
    return {
        "scope": (
            "private local resident inference: transform+backbone+adapter+optional subtype scores and"
            " reporting joint probabilities+hierarchical_decision; excludes checkpoint loading, startup,"
            " network, audit"
        ),
        "device": device,
        "batch_size": batch_size,
        "validation_rows": len(raw),
        "batch_repetitions": 3,
        "batch_total_ms": batch_ms,
        "batch_total_ms_median": float(np.median(batch_ms)),
        "batch_ms_per_row_median": float(np.median(batch_ms)) / len(raw),
        "single_request_repetitions": 20,
        "single_request_input": "one fixed local VAL row; timing variability only",
        "single_request_ms_median": float(np.median(request_ms)),
        "single_request_ms_p95_observed": float(np.quantile(request_ms, 0.95)),
        "max_probability_abs_diff_vs_cached": difference,
        "includes_transform_adapter_and_decision": True,
    }


def execute_secure(args):
    # Import lazily to keep one shared detector/data/decision implementation.
    from .privacy_run import (
        DATASETS,
        ROOT,
        PrivateClient,
        client_partitions,
        risk_metrics,
        synthetic_data,
    )

    if args.clients < 3:
        raise ValueError("MPC requires at least three clients and the complete fixed roster")
    if not math.isfinite(args.clip) or args.clip <= 0:
        raise ValueError("Positive finite secure update clipping bound required")
    if type(args.secure_scale) is not int or args.secure_scale < 1:
        raise ValueError("Positive integer secure fixed-point scale required")
    if not math.isfinite(args.secure_timeout) or args.secure_timeout <= 0:
        raise ValueError("Positive finite secure process timeout required")
    if args.ledger_dir is not None:
        raise ValueError("--ledger-dir is DP-only; MPC never opens or resets DP ledgers")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if args.out is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        args.out = str(ROOT / "outputs" / f"secure_{stamp}_{uuid.uuid4().hex[:6]}")
    root = Path(args.out).resolve()
    if root.exists():
        raise ValueError("Use a NEW --out; existing experiments are never overwritten or auto-resumed")
    root.mkdir(parents=True)
    public, private = root / "public", root / "private_DO_NOT_UPLOAD"
    public.mkdir()
    private.mkdir()
    session = uuid.uuid4().hex
    from .experiment_config import (
        load_local_for_dataset,
        non_numerical_option,
        numerical_parameters,
        resolve_dataset_parameters,
        selected_datasets,
    )

    selected = selected_datasets(args)
    settings = {
        k: v
        for k, v in vars(args).items()
        if not non_numerical_option(k)
        and k
        not in (
            "out",
            "data_root",
            "ledger_dir",
            "improve_from",
            "sigma",
            "delta",
            "max_epsilon",
            "gate_family_error",
            "subtype_cap",
            "audit_data",
        )
    }
    core.save_json(
        public / "protocol.json",
        {
            "session": session,
            "settings": settings,
            "framework_version": 3,
            "shared_algorithm": (
                "prior_adjusted_PLE_TabM_class_score_secure_sum_personalization_hierarchical_alert"
            ),
            "dataset_parameters": {name: numerical_parameters(args, name) for name in selected},
            "datasets": list(selected),
            "adapter_dimension_rule": "public_class_count_minus_one",
            "mechanism": "two_server_additive_secure_sum",
            "differential_privacy": False,
            "epsilon": None,
            "delta": None,
            "test_opened": False,
            "privacy_scope": (
                "individual shares hidden from each separate honest-but-curious noncolluding collector;"
                " aggregate is revealed"
            ),
            "trust_assumption": (
                "two collectors do not collude; trusted local simulation orchestrator; honest client clipping"
            ),
            "deployment": (
                "two subprocess collectors on one trusted host; no TLS, host-admin isolation or distributed"
                " deployment"
            ),
            "audit": "local hash-chain audit; no signatures, consensus or blockchain deployment",
            "historical_dp_ledgers_modified": False,
            "combined_transcript_has_dp_guarantee": False,
            "raw_data_or_full_models_uploaded": False,
            "private_diagnostics_are_dp": False,
        },
    )
    from ._iov_layout import SNAPSHOT_SUPPORT_FILES, source_path

    snapshot = root / "execution_source"
    snapshot.mkdir()
    files = (
        "experiment_config.py",
        "data_io.py",
        "privacy_core.py",
        "privacy_run.py",
        "secure_run.py",
        "secure_aggregation.py",
        "aggregation_audit.py",
        "validation_core.py",
        "vendor/tabm.py",
        "vendor/rtdl_num_embeddings.py",
    ) + SNAPSHOT_SUPPORT_FILES
    pins = {}
    for relative in dict.fromkeys(files):
        saved = snapshot / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(relative), saved)
        pins[relative] = core.sha(saved)
    core.save_json(public / "execution_source_hashes.json", pins)
    core.save_json(
        private / "environment_private.json",
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "partition_seed": args.partition_seed,
            "training_seed": args.training_seed,
        },
    )
    report, validation_rows = [], []
    shared_args = args
    for name in selected:
        args = resolve_dataset_parameters(shared_args, name)
        dataset_started = time.perf_counter()
        from .round_observability import ResourceSampler, observe_rounds

        sampler = ResourceSampler().start() if getattr(args, "_observe_rounds", False) else None
        try:
            clients = [f"client_{i}" for i in range(args.clients)]
            load_started = time.perf_counter()
            parts, names, features = (
                synthetic_data(DATASETS.index(name)) if args.smoke else load_local_for_dataset(args, name)
            )
            partitions = client_partitions(
                parts,
                len(names),
                args.clients,
                args.alpha,
                args.partition_seed,
                args.rows_per_client,
            )
            dimension = len(names) - 1
            load_seconds = time.perf_counter() - load_started
            nodes = []
            for client, rows in zip(clients, partitions):
                print(
                    f"LOCAL FIT {name}/{client}: training_seed={args.training_seed}; starting private"
                    " detector training",
                    flush=True,
                )
                node = PrivateClient(client, parts, rows, names, features, args, private / name / client)
                node.framework_version, node.mechanism = 3, "mpc"
                # Cache private probabilities to avoid repeated backbone inference.
                np.savez_compressed(
                    node.directory / "prediction_cache_private.npz",
                    cal_probabilities=node.probabilities["cal"],
                    cal_labels=node.parts["cal"][1],
                    val_probabilities=node.probabilities["val"],
                    val_labels=node.parts["val"][1],
                )
                core.save_json(
                    node.directory / "prediction_cache_manifest_private.json",
                    {
                        "schema": 1,
                        "class_names": names,
                        "feature_names": features,
                        "partition_seed": args.partition_seed,
                        "training_seed": args.training_seed,
                        "backbone_state_sha256": _model_hash(node.model),
                        "transform_sha256": digest(node.transform.state()),
                        "partition_indices_sha256": {
                            role: hashlib.sha256(rows[role].tobytes()).hexdigest() for role in rows
                        },
                        "execution_source_sha256": {
                            k: core.sha(snapshot / k)
                            for k in (
                                "utils.py",
                                "privacy_run.py",
                                "experiment_config.py",
                                "dataset_access.py",
                            )
                        },
                        "cache_sha256": core.sha(node.directory / "prediction_cache_private.npz"),
                        "scope": "local CAL/VAL base probabilities; private; no TEST",
                    },
                )
                nodes.append(node)
            print(
                f"SECURE FEDERATION {name}: detectors fitted; adapter dimension={dimension}",
                flush=True,
            )
            data_public = public / name
            data_public.mkdir()
            audit = AggregationAudit(
                data_public / "audit.jsonl", name, clients, dimension, args.clip, args.secure_scale
            )
            global_vector = np.zeros(dimension)
            adapter = ClassScoreAdapter(len(names))
            rounds = []
            round_timings, global_snapshots = [], []
            audit_seconds, aggregation_seconds, serialization_seconds = 0.0, 0.0, 0.0
            process_started = time.perf_counter()
            with SecureAggregationSession(
                clients,
                dimension,
                clip=args.clip,
                scale=args.secure_scale,
                timeout=args.secure_timeout,
            ) as channel:
                process_startup_seconds = time.perf_counter() - process_started
                for round_index in range(1, args.rounds + 1):
                    resource_token = sampler.begin_interval() if sampler else None
                    fit_seconds_before = sum(
                        n.performance.get("adapter_training_seconds", 0.0) for n in nodes
                    )
                    aggregation_before = aggregation_seconds
                    parent = digest(global_vector.tolist())
                    contract = channel.start_round(
                        session=session, round_index=round_index, parent_model=parent
                    )
                    audit_started = time.perf_counter()
                    audit.reserve(contract)  # durable contract before any share submission
                    audit_seconds += time.perf_counter() - audit_started
                    for node in nodes:
                        node.submit_secure(channel, global_vector, contract)
                    aggregate_started = time.perf_counter()
                    mean, receipt = channel.aggregate()  # never sees an individual unmasked delta
                    previous = global_vector
                    global_vector = adapter.project(global_vector + mean)
                    aggregation_seconds += time.perf_counter() - aggregate_started
                    receipt = {
                        **receipt,
                        "parent_model": parent,
                        "model_hash": digest(global_vector.tolist()),
                        "round": round_index,
                        "accepted_update": bool(np.any(global_vector != previous)),
                        "adapter_schema": "class_logit_bias_v2",
                        "differential_privacy": False,
                    }
                    audit_started = time.perf_counter()
                    audit_state = audit.commit(contract, global_vector, receipt)
                    audit_seconds += time.perf_counter() - audit_started
                    serialization_started = time.perf_counter()
                    core.save_json(
                        data_public / f"r{round_index}_aggregate.json",
                        {
                            **receipt,
                            "audit_head": audit_state["head"],
                            "adapter": global_vector.tolist(),
                        },
                    )
                    serialization_seconds += time.perf_counter() - serialization_started
                    rounds.append(receipt)
                    global_snapshots.append({"round": round_index, "adapter": global_vector.copy().tolist()})
                    round_timings.append({
                        "round": round_index,
                        "adapter_fit_seconds": (
                            sum(n.performance.get("adapter_training_seconds", 0.0) for n in nodes)
                            - fit_seconds_before
                        ),
                        "aggregation_seconds": aggregation_seconds - aggregation_before,
                        "resources": sampler.end_interval(resource_token) if sampler else None,
                    })
            diagnostics = [
                {
                    "client": node.client_id,
                    **node.save_private(global_vector),
                    "detector_parameters": node.parameter_count,
                }
                for node in nodes
            ]
            benchmark_started = time.perf_counter()
            for node in nodes:
                benchmark = _benchmark_inference(
                    node, parts["val"][0][node.rows["val"]], args.device, args.batch_size
                )
                core.save_json(node.directory / "end_to_end_inference_private.json", benchmark)
            inference_benchmark_seconds = time.perf_counter() - benchmark_started
            core.save_json(private / name / "validation_private.json", diagnostics)
            pooled = risk_metrics(
                np.concatenate([n.parts["val"][1] for n in nodes]),
                np.concatenate([n.final_probabilities for n in nodes]),
                names,
                np.concatenate([np.full(len(n.parts["val"][1]), n.alarm_threshold) for n in nodes]),
            )
            core.save_json(private / name / "pooled_validation_private.json", pooled)
            numeric_upload = 2 * args.rounds * args.clients * dimension * 8
            numeric_aggregate = 2 * args.rounds * dimension * 8
            numeric_download = args.rounds * args.clients * dimension * 8
            public_json = sum(p.stat().st_size for p in data_public.glob("r*_aggregate.json")) * args.clients
            privacy_metrics = {
                "mechanism": "two_server_additive_secure_sum",
                "privacy_guarantee": "secure_sum_no_DP",
                "differential_privacy": False,
                "epsilon": None,
                "delta": None,
                "privacy_scope": "per-collector share confidentiality; final aggregate is revealed",
                "collectors": 2,
                "minimum_clients": 3,
                "complete_roster_required": True,
                "fixed_point_scale": args.secure_scale,
                "mean_coordinate_quantization_error_bound": 0.5 / args.secure_scale,
                "update_clipping_l2_bound": args.clip,
                "individual_update_publications": 0,
                "numeric_share_upload_bytes": numeric_upload,
                "numeric_collector_sum_bytes": numeric_aggregate,
                "numeric_adapter_download_bytes": numeric_download,
                "numeric_payload_bytes": numeric_upload + numeric_aggregate + numeric_download,
                "aggregate_json_logical_download_bytes": public_json,
                "byte_scope": (
                    "numeric int64 shares and float64 adapters; aggregate JSON downloads counted separately;"
                    " no TCP/TLS/consensus claim"
                ),
                "completed_rounds": args.rounds,
                "accepted_updates": sum(r["accepted_update"] for r in rounds),
                "federated_adaptation_applied": any(r["accepted_update"] for r in rounds),
                "adapter_schema": "class_logit_bias_v2",
            }
            # Publish protocol-message sizes, never individual shares or update norms.
            for field in ("local_ipc_frame_bytes", "serialized_application_bytes"):
                if all(field in r for r in rounds):
                    privacy_metrics[field] = sum(r[field] for r in rounds)
            performance = {
                "data_load_and_partition_seconds": load_seconds,
                "secure_process_startup_seconds": process_startup_seconds,
                "audit_seconds": audit_seconds,
                "server_aggregation_seconds": aggregation_seconds,
                "message_serialization_seconds": serialization_seconds,
                "end_to_end_inference_benchmark_seconds": inference_benchmark_seconds,
                "clients": [{"client": n.client_id, **n.performance} for n in nodes],
                "total_experiment_seconds": time.perf_counter() - dataset_started,
                "raw_train_rows": len(parts["train"][1]),
                "selected_train_rows": sum(len(n.parts["train"][1]) for n in nodes),
                "full_assigned_train": args.rows_per_client == 0,
                "detector_parameters": sum(n.parameter_count for n in nodes),
                "optimizer_steps": sum(n.performance["optimizer_steps"] for n in nodes),
                "training_row_visits": sum(n.performance["training_row_visits"] for n in nodes),
                "model_checkpoint_bytes": sum(
                    (n.directory / "detector_private.pt").stat().st_size for n in nodes
                ),
                "timing_scope": (
                    "trusted local simulator; excludes production network and blockchain consensus"
                ),
            }
            performance["secure_processing_seconds"] = (
                process_startup_seconds
                + audit_seconds
                + aggregation_seconds
                + sum(n.performance.get("secure_submission_seconds", 0.0) for n in nodes)
            )
            performance["secure_processing_scope"] = (
                "recorded startup, audit, client encoding/submission and server aggregation "
                "subtotal; excludes round-begin IPC, worker shutdown, local adapter fitting "
                "and result-file serialization"
            )
            performance["secure_processing_fraction"] = performance["secure_processing_seconds"] / max(
                performance["total_experiment_seconds"], 1e-12
            )
            core.save_json(private / name / "performance_private.json", performance)
            if sampler:
                resources = sampler.stop()
                core.save_json(private / name / "resource_observations_private.json", resources)
                observe_rounds(
                    nodes,
                    global_snapshots,
                    private_dir=private / name,
                    dataset=name,
                    training_seed=args.training_seed,
                    partition_seed=args.partition_seed,
                    original_round_timings=round_timings,
                    resources_summary=resources,
                )
            validation_rows.append({
                "dataset": name,
                **{
                    k: pooled[k]
                    for k in (
                        "accuracy",
                        "weighted_f1",
                        "macro_f1",
                        "far",
                        "fnr",
                        "argmax_accuracy",
                        "argmax_weighted_f1",
                        "argmax_macro_f1",
                        "binary_brier",
                        "binary_ece_10",
                        "binary_nll",
                    )
                },
                "total_experiment_seconds": performance["total_experiment_seconds"],
                "secure_processing_seconds": performance["secure_processing_seconds"],
            })
            _write_csv(private / "detection_metrics_private.csv", validation_rows)
            report.append({
                "dataset": name,
                "domain": "in_vehicle_CAN" if name in DATASETS[:2] else "external_network_proxy",
                "mechanism": "mpc",
                "framework_version": 3,
                "clients": args.clients,
                "adapter_dimension": dimension,
                "completed_rounds": args.rounds,
                "privacy_metrics": privacy_metrics,
                "rounds": rounds,
                "audit": audit.verify(),
                "synthetic": args.smoke,
            })
            core.save_json(public / "summary.json", report)
            _write_csv(
                public / "privacy_metrics.csv",
                [{"dataset": r["dataset"], **r["privacy_metrics"]} for r in report],
            )
            print(
                f"SECURE FEDERATION {name}: {args.rounds} complete-roster rounds; "
                f"nonzero_update_rounds={privacy_metrics['accepted_updates']}; "
                "MPC confidentiality, no DP epsilon",
                flush=True,
            )
            del nodes, node, parts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if sampler:
                sampler.stop()
    args = shared_args
    core.save_json(private / "privacy_validation_private.json", privacy_scope_validation(public))
    core.save_json(
        private / "experiment_validation_private.json",
        {
            "CAL_fit_threshold_disjoint": True,
            "VAL_used_for_training": False,
            "independent_TEST_evaluated": False,
            "statistical_superiority_established": False,
            "unknown_data_non_degradation_guaranteed": False,
            "scope": "fresh v3 training; development validation, no reference release gate",
        },
    )
    core.save_json(
        public / "completion.json",
        {
            "complete": True,
            "synthetic": args.smoke,
            "framework_version": 3,
            "mechanism": "mpc",
            "datasets": list(selected),
            "differential_privacy": False,
            "epsilon": None,
            "delta": None,
            "test_opened": False,
            "blockchain_deployed": False,
            "research_superiority_established": False,
            "private_metrics": "../private_DO_NOT_UPLOAD (outside public transcript)",
        },
    )
    print(f"COMPLETE: {len(report)} independent secure federations. Results: {root}", flush=True)
    return 0
