"""Measure synthetic updates on local or explicitly declared TCP collectors."""

import csv
from ipaddress import ip_address
from pathlib import Path
from tempfile import TemporaryDirectory

from . import revision_protocol_validation
from .revision_local_transport import CLIENTS, DIMENSIONS, LOOPBACK, METHOD_SOURCES, ROUNDS
from .revision_local_transport import summarize_loopback_dimension
from .revision_v12 import _bindings, _hash, _output_lock, _read, _write


HOST_ROLES = ("orchestrator", "collector_a", "collector_b")
HOST_FIELDS = ("physical_host_id", "os", "cpu", "ram")
MEASUREMENT_FILES = ("protocol_summary.json", "protocol_rounds.csv")
LOOPBACK_SUMMARY = "loopback_summary.csv"
LOOPBACK_PLAN = "loopback_plan.json"
LOOPBACK_RESULT = "complete.json"
LOOPBACK_TIMEOUT = 30.0
LOOPBACK_WORKLOAD = {"seed": 2026, "clip": 0.1, "scale": 1_000_000}


def _host_identity(host, role):
    for field in HOST_FIELDS:
        value = host.get(field)
        if not isinstance(value, str) or not value.strip() or value.strip().lower().startswith("replace"):
            raise ValueError(f"Record physical_host_id, os, cpu and ram for {role}")
    return host["physical_host_id"].strip().casefold()


def _collector_endpoint(host, role):
    address, port = host["endpoint"].rsplit(":", 1)
    ip = ip_address(address)
    if (
        ip.version != 4
        or not ip.is_private
        or ip.is_loopback
        or ip.is_unspecified
        or not 1 <= int(port) <= 65535
    ):
        raise ValueError(f"Use a private, non-loopback IPv4 LAN endpoint for {role}")
    return f"{ip}:{int(port)}"


def _declared_endpoints(config):
    if config.get("version") != 1 or not str(config.get("topology", "")).strip():
        raise ValueError("LAN configuration requires version 1 and a topology description")
    hosts = [_host_identity(config[role], role) for role in HOST_ROLES]
    endpoints = [_collector_endpoint(config[role], role) for role in HOST_ROLES[1:]]
    if len(set(hosts)) != 3 or endpoints[0].split(":")[0] == endpoints[1].split(":")[0]:
        raise ValueError("Declare three distinct physical hosts and two distinct collector addresses")
    return endpoints


def preflight_lan(manifest):
    """Validate the LAN declaration locally and return readiness metadata."""
    manifest = Path(manifest).resolve()
    if manifest.name.endswith(".example.json") or not manifest.is_file():
        return {
            "status": "pending",
            "reason": "LAN measurement configuration is missing",
            "manifest": str(manifest),
        }
    config = _read(manifest)
    return {
        "status": "ready",
        "manifest": str(manifest),
        "manifest_sha256": _hash(manifest),
        "operator_declared_topology": config,
        "endpoints": _declared_endpoints(config),
        "independent_physical_hosts_verified": False,
    }


def _save_plan(path, ready):
    protocol = Path(__file__).with_name("revision_protocol_validation.py")
    plan = {
        "setup": ready,
        "clients": list(CLIENTS),
        "dimensions": list(DIMENSIONS),
        "rounds": ROUNDS,
        "protocol_source_sha256": _hash(protocol),
    }
    if path.exists() and _read(path) != plan:
        raise ValueError("Saved LAN plan differs; use a separate output directory for a different topology")
    _write(path, plan)


def _measure(destination, endpoints, dimension):
    with TemporaryDirectory(prefix="lan_measurement_", dir=destination.parent) as temporary:
        stage = Path(temporary) / "results"
        revision_protocol_validation.run_protocol_validation(
            stage,
            clients=CLIENTS,
            dimension=dimension,
            rounds=ROUNDS,
            timeout=30.0,
            collector_a=endpoints[0],
            collector_b=endpoints[1],
        )
        stage.rename(destination)


def _check_measurement(destination, endpoints, dimension):
    summary = _read(destination / MEASUREMENT_FILES[0])
    conditions = [
        (item["clients"], item["measured_rounds"], item["all_plaintext_quantized_sums_equal"])
        for item in summary["conditions"]
    ]
    if (
        summary.get("complete") is not True
        or summary.get("transport_environment") != "user_supplied_endpoints"
        or summary.get("independent_physical_hosts_verified") is not False
        or summary.get("endpoints") != endpoints
        or summary.get("dimension") != dimension
        or conditions != [(count, ROUNDS, True) for count in CLIENTS]
    ):
        raise ValueError(f"LAN receipt differs from the plan: {destination}")


def _dimension_files(output, dimension, endpoints, recorded):
    destination = output / f"dimension_{dimension}"
    if destination.resolve() != destination:
        raise ValueError(f"Redirected LAN output: {destination}")
    if recorded is not None:
        if _bindings(destination, MEASUREMENT_FILES) != recorded:
            raise ValueError(f"Saved LAN artifacts changed: {destination}")
    elif not destination.exists():
        _measure(destination, endpoints, dimension)
    _check_measurement(destination, endpoints, dimension)
    return _bindings(destination, MEASUREMENT_FILES)


def _save_receipt(output, ready, plan_path, state_path, files):
    receipt = {
        "status": "completed",
        "output": str(output),
        "transport_environment": "user_supplied_endpoints",
        "independent_physical_hosts_verified": False,
        "detector_training_measured": False,
        "inference_measured": False,
        "operator_declared_topology": ready["operator_declared_topology"],
        "plan_sha256": _hash(plan_path),
        "state_sha256": _hash(state_path),
        "rounds_measured": len(CLIENTS) * len(DIMENSIONS) * ROUNDS,
        "artifacts": {
            f"dimension_{role}/{path}": sha for role, part in files.items() for path, sha in part.items()
        },
    }
    _write(output / "complete.json", receipt)
    return receipt


def run_lan(output, manifest):
    """Run the declared LAN measurements and preserve each completed result."""
    output = Path(output).resolve()
    ready = preflight_lan(manifest)
    if ready["status"] != "ready":
        return {"output": str(output), **ready}
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        plan_path = output / "lan_plan.json"
        state_path = output / "lan_state.json"
        _save_plan(plan_path, ready)
        files = _read(state_path) if state_path.exists() else {}
        for dimension in DIMENSIONS:
            role = str(dimension)
            files[role] = _dimension_files(output, dimension, ready["endpoints"], files.get(role))
            _write(state_path, files)
        return _save_receipt(output, ready, plan_path, state_path, files)


def _loopback_code_hashes():
    """Bind the benchmark, aggregation and reporting sources from this run."""
    source = Path(__file__).resolve().parent
    return _bindings(source, (*METHOD_SOURCES, "revision_local_transport.py", Path(__file__).name))


def _loopback_plan():
    return {
        "schema_version": 1,
        "transport_environment": LOOPBACK,
        "bound_collector_host": "127.0.0.1",
        "clients": list(CLIENTS),
        "dimensions": list(DIMENSIONS),
        "rounds_per_condition": ROUNDS,
        "timeout_seconds": LOOPBACK_TIMEOUT,
        "workload": LOOPBACK_WORKLOAD,
        "source_sha256": _loopback_code_hashes(),
    }


def _measure_loopback(output, dimension):
    """Run a real native synthetic workload against new localhost collectors."""
    directory = output / f"dimension_{dimension}"
    revision_protocol_validation.run_protocol_validation(
        directory,
        clients=CLIENTS,
        dimension=dimension,
        rounds=ROUNDS,
        timeout=LOOPBACK_TIMEOUT,
        collector_a=None,
        collector_b=None,
        **LOOPBACK_WORKLOAD,
    )
    _, summary, _ = summarize_loopback_dimension(directory, dimension, workload=LOOPBACK_WORKLOAD)
    names = [f"{directory.name}/{name}" for name in MEASUREMENT_FILES]
    return summary, names


def _loopback_receipt(output, plan, rows, artifacts):
    """State the timing and security scope of this same-host microbenchmark."""
    return {
        "schema_version": 1,
        "status": "completed",
        "output": str(output),
        "evidence_origin": "current_fresh_protocol_measurement",
        "measurements_reexecuted": True,
        "transport_environment": LOOPBACK,
        "independent_physical_hosts_verified": False,
        "detector_training_measured": False,
        "inference_measured": False,
        "synthetic_updates": True,
        "startup_in_round_times": False,
        "clients": plan["clients"],
        "dimensions": plan["dimensions"],
        "rounds_per_condition": ROUNDS,
        "conditions_measured": len(rows),
        "rounds_measured": sum(row["measured_rounds"] for row in rows),
        "all_plaintext_quantized_sums_equal": all(row["all_plaintext_quantized_sums_equal"] for row in rows),
        "all_clients_accepted": all(row["all_clients_accepted"] for row in rows),
        "scope": "Same-host TCP round timing includes clipping, sharing, exchanges and reconstruction checks",
        "numeric_payload_scope": "uint64 client and aggregate shares without framing",
        "tcp_application_byte_scope": "JSON and frame prefixes; excludes TCP/IP headers and retransmissions",
        "security_scope": "Trusted orchestrator, two noncolluding honest-but-curious collectors",
        "security_exclusions": [
            "TLS",
            "authentication",
            "DP",
            "malicious-party protection",
            "dropout recovery",
        ],
        "source_sha256": plan["source_sha256"],
        "artifacts": artifacts,
    }


def run_loopback(output):
    """Measure a fresh localhost matrix and publish its bound evidence."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = _loopback_plan()
    _write(output / LOOPBACK_PLAN, plan)
    rows, names = [], [LOOPBACK_PLAN, LOOPBACK_SUMMARY]
    for dimension in DIMENSIONS:
        table, artifacts = _measure_loopback(output, dimension)
        rows.extend(table)
        names.extend(artifacts)
    if _loopback_code_hashes() != plan["source_sha256"]:
        raise ValueError("Loopback benchmark source changed during measurement")
    with (output / LOOPBACK_SUMMARY).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = _loopback_receipt(output, plan, rows, _bindings(output, names))
    _write(output / LOOPBACK_RESULT, receipt)
    return {**receipt, "artifacts": {**receipt["artifacts"], **_bindings(output, (LOOPBACK_RESULT,))}}
