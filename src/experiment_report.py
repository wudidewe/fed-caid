"""Compact, read-only presentation of completed local experiment JSON files."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import uuid

from experiment_report_data import load_runs, ReportDataError


PRIMARY = ("accuracy", "weighted_f1", "far", "fnr")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "weighted_f1": "F1",
    "far": "False alarm rate",
    "fnr": "False negative rate",
}
CONFUSION_DATASETS = ("CIC-IDS-2017", "CIC-UNSW-NB15")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat_text(stat, percentage=True):
    factor = 100 if percentage else 1
    return f'{factor * stat["mean"]:.3f}'


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, folder, name, figures):
    for suffix in ("png", "pdf"):
        path = folder / (name + "." + suffix)
        fig.savefig(path, dpi=600, facecolor="white")
    figures.append(name)


CLASS_ALIASES = {
    "BENIGN": "Benign",
    "DoS GoldenEye": "GoldenEye",
    "DoS Hulk": "Hulk",
    "DoS Slowhttptest": "SlowHTTP",
    "DoS slowloris": "Slowloris",
    "FTP-Patator": "FTP",
    "SSH-Patator": "SSH",
    "Web Attack - Brute Force": "Web brute",
    "Web Attack - Sql Injection": "SQLi",
    "Web Attack - XSS": "XSS",
    "Reconnaissance": "Recon",
}
CONFUSION_FILES = {
    "CIC-IDS-2017": "confusion_cicids2017",
    "CIC-UNSW-NB15": "confusion_cicunsw_nb15",
}


def _communication_totals(rounds, key):
    """Missing byte counters are unknown, never a measured zero."""
    counters = [row["communication"] for row in rounds]
    return sum(row[key] for row in counters) if all(key in row for row in counters) else None


def _prediction_measurement(data, name):
    """Amortized CAL+VAL base prediction, not isolated request latency."""
    clients = data["runs"][0]["datasets"][name]["timing"]["clients"]
    seconds = [client["values"].get("inference_seconds") for client in clients]
    rows = [client.get("inference_rows") for client in clients]
    available = all(value is not None for value in seconds + rows)
    return {
        "seconds_sum_clients": sum(seconds) if all(v is not None for v in seconds) else None,
        "rows_sum_clients": sum(rows) if all(v is not None for v in rows) else None,
        "microseconds_per_record": 1e6 * sum(seconds) / sum(rows) if available else None,
        "scope": "Amortized base prediction over recorded CAL plus VAL batches: summed client seconds divided by summed client records. Excludes full calibrated decisions and is not isolated request latency.",
    }


def _plot_scope(data):
    return {
        name: {
            "training_seconds_sum_clients": data["summary"][name]["timing"]["metrics"]
            .get("detector_training_seconds_sum_clients", {})
            .get("mean"),
            "prediction": _prediction_measurement(data, name),
            "communication": {
                key: _communication_totals(data["summary"][name]["aggregation"][0]["rounds"], key)
                for key in ("numeric_share_upload_bytes", "local_ipc_frame_bytes")
            },
        }
        for name in data["datasets"]
    }


def _missing(axis):
    axis.text(
        0.5,
        0.5,
        "Not recorded",
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#455564",
    )


def _plots(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures, folder = [], output / "figures"
    folder.mkdir()
    names = data["datasets"]
    for name, filename in CONFUSION_FILES.items():
        if name not in names:
            continue
        matrix = np.asarray(data["summary"][name]["confusion"]["matrix"], dtype=float)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(100 * matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        labels = [CLASS_ALIASES.get(label, label) for label in data["summary"][name]["class_names"]]
        fig = plt.figure(figsize=(8, 7.4))
        axis = fig.add_axes([0.17, 0.20, 0.70, 5.6 / 7.4])
        picture = axis.imshow(
            normalized, cmap="Blues", vmin=0, vmax=100, interpolation="nearest", aspect="equal"
        )
        axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=12)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=12)
        axis.set_xlabel("Predicted class", labelpad=8)
        axis.set_ylabel("True class", labelpad=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = normalized[i, j]
                if matrix[i, j] > 0 or i == j:
                    label = "<0.1" if 0 < value < 0.1 else f"{value:.1f}".removesuffix(".0")
                    axis.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        color="white" if value > 55 else "#202020",
                    )
        color_axis = fig.add_axes([0.888, 0.20, 0.020, 5.6 / 7.4])
        colorbar = fig.colorbar(picture, cax=color_axis)
        colorbar.set_label("Row percent", labelpad=5)
        colorbar.ax.tick_params(labelsize=11)
        _save(fig, folder, filename, figures)
        plt.close(fig)
    measurements = _plot_scope(data)
    x = np.arange(len(names))
    for filename, ylabel, values in (
        (
            "training_time",
            "Detector training time (s)",
            [measurements[name]["training_seconds_sum_clients"] for name in names],
        ),
        (
            "prediction_time",
            "Base prediction time (µs / record)",
            [measurements[name]["prediction"]["microseconds_per_record"] for name in names],
        ),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.3))
        fig.subplots_adjust(left=0.13, bottom=0.30, right=0.97, top=0.95)
        valid = [(index, value) for index, value in enumerate(values) if value is not None]
        if valid:
            axis.bar(
                [index for index, _ in valid],
                [value for _, value in valid],
                width=0.58,
                color="#27689e",
            )
        else:
            _missing(axis)
        axis.set_xticks(x, names, rotation=20, ha="right", fontsize=11)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        _save(fig, folder, filename, figures)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.13, bottom=0.30, right=0.97, top=0.95)
    plotted = []
    for key, offset, label, color in (
        ("numeric_share_upload_bytes", -0.19, "Numeric client shares", "#27689e"),
        ("local_ipc_frame_bytes", 0.19, "Round IPC commands and replies", "#9cb5c8"),
    ):
        valid = [
            (index, measurements[name]["communication"][key])
            for index, name in enumerate(names)
            if measurements[name]["communication"][key] is not None
        ]
        if valid:
            axis.bar(
                [index + offset for index, _ in valid],
                [value for _, value in valid],
                width=0.36,
                color=color,
                label=label,
            )
            plotted.extend(value for _, value in valid)
    if plotted:
        axis.legend(frameon=False, fontsize=11, loc="upper left")
        axis.set_ylim(0, max(plotted) * 1.28 or 1)
    else:
        _missing(axis)
    axis.set_xticks(x, names, rotation=20, ha="right", fontsize=11)
    axis.set_ylabel("Recorded communication (bytes)")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    _save(fig, folder, "communication_volume", figures)
    plt.close(fig)
    return figures


def _html(data, output, figures):
    label = "SYNTHETIC FUNCTIONAL CHECK" if data["synthetic"] else "DEVELOPMENT RESULTS"
    headers = "<th>Dataset</th>" + "".join(f"<th>{METRIC_LABELS[k]}</th>" for k in PRIMARY)
    rows = []
    rare = []
    for name in data["datasets"]:
        summary = data["summary"][name]
        values = "".join(f'<td>{escape(_stat_text(summary["metrics"][k]))}</td>' for k in PRIMARY)
        rows.append(f"<tr><th>{escape(name)}</th>{values}</tr>")
        weakest = sorted(summary["per_class"].items(), key=lambda item: item[1]["f1"]["mean"])[:2]
        rare.append(
            "<tr><th>"
            + escape(name)
            + "</th><td>"
            + "; ".join(
                f'{escape(cls)}: recall {100 * value["recall"]["mean"]:.2f}%, n={value["support"]}'
                for cls, value in weakest
            )
            + "</td></tr>"
        )
    pictures = []
    captions = {
        "confusion_cicids2017": "Class decisions on CIC-IDS-2017.",
        "confusion_cicunsw_nb15": "Class decisions on CIC-UNSW-NB15.",
        "training_time": "Local detector training time.",
        "prediction_time": "Amortized base prediction time.",
        "communication_volume": "Recorded aggregation communication.",
    }
    descriptions = {
        "confusion_cicids2017": "Rows show percentages from recorded integer confusion counts. All classes are retained. Positive percentages below 0.1 appear as <0.1; diagonal zeros remain visible. FTP and SSH denote Patator attacks, SQLi denotes SQL Injection, and Web brute denotes Web Attack Brute Force.",
        "confusion_cicunsw_nb15": "Rows show percentages from recorded integer confusion counts. Positive percentages below 0.1 appear as <0.1; diagonal zeros remain visible. Recon denotes Reconnaissance. Both confusion figures have identical canvases and physical matrix sizes.",
        "training_time": "Recorded detector training seconds summed over clients. This excludes preprocessing, calibration and aggregation and is not a complete workflow stopwatch. Missing observations are omitted.",
        "prediction_time": "One million times summed client inference_seconds divided by summed recorded inference_rows. The base prediction timer covers CAL plus VAL batches. This is amortized batch processing, not isolated request latency or complete calibrated decisions. Missing seconds or row counts stay unavailable.",
        "communication_volume": "Counters are summed over completed aggregation rounds. Numeric client shares count numerical uploads to both collectors. Round IPC counts serialized commands and replies. These scopes differ and are not additive; startup and operating system transport overhead are excluded.",
    }
    for name in figures:
        encoded = base64.b64encode((output / "figures" / (name + ".png")).read_bytes()).decode(
            "ascii"
        )
        pictures.append(
            f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(captions[name])}">'
            f"<figcaption>{escape(captions[name])}</figcaption><p>{escape(descriptions[name])}</p></figure>"
        )
    rounds = []
    for name in data["datasets"]:
        for a in data["summary"][name]["aggregation"]:
            rounds.append(
                f'<tr><th>{escape(name)}</th><td>{a["training_seed"]}</td>'
                f'<td>{a["completed_rounds"]}</td><td>{escape(str(a["all_clients_accepted"]))}</td>'
                f'<td>{a["nonzero_update_rounds"]}</td></tr>'
            )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedCAID experiment report</title><style>
body{font:15px/1.5 Arial,sans-serif;max-width:1160px;margin:28px auto;padding:0 20px;color:#202c39;background:#fff}
h1{font-size:25px;margin-bottom:6px}h2{font-size:19px;margin-top:24px}p{margin:7px 0}.tag{font-weight:bold;color:#27689e}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}th,td{border:1px solid #d6dde3;padding:8px;text-align:right}
th:first-child{text-align:left}thead{background:#edf2f6}figure{margin:18px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#455564}
details{margin:18px 0}footer{font-size:12px;color:#586575;margin:24px 0}a{color:#27689e}
</style>"""
    html += f'<h1>FedCAID experiment report</h1><p class="tag">{label}</p>'
    html += "<p>VAL development evaluation. Independent TEST was not evaluated. F1 denotes support-weighted F1.</p>"
    if data["framework_version"] == 5:
        choices = [
            run["datasets"][name]["selection"]["arm"]
            for run in data["runs"]
            for name in data["datasets"]
        ]
        html += f'<p>Final selected results: reference retained in {choices.count("reference")}/{len(choices)} federations; candidate selected in {choices.count("candidate")}/{len(choices)}.</p>'
    html += (
        "<h2>Detection summary (%)</h2><table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    html += (
        "<details><summary>Class support and aggregation status</summary><table><thead><tr><th>Dataset</th><th>Lowest F1 classes and recall</th></tr></thead>"
        + "".join(rare)
        + "</table>"
    )
    html += (
        "<table><thead><tr><th>Dataset</th><th>Seed</th><th>Completed rounds</th><th>All clients submitted</th><th>Nonzero updates</th></tr></thead>"
        + "".join(rounds)
        + "</table></details>"
    )
    html += "<details><summary>Experiment figures</summary>" + "".join(pictures) + "</details>"
    html += "<footer>Private local report. Default MPC has no DP guarantee. No datasets, models, training or original results were modified. Figures are exported separately in PNG and vector PDF formats. Configuration and source hashes are recorded in manifest.json.</footer></html>"
    (output / "report.html").write_text(html, encoding="utf-8")


def generate_report(paths, output):
    roots = [Path(p).resolve() for p in paths]
    output = Path(output).resolve()
    if output.exists():
        raise ReportDataError(
            "Use a NEW report output directory; existing files are never overwritten"
        )
    if any(output.is_relative_to(root) for root in roots):
        raise ReportDataError("Write reports outside the original experiment directories")
    data = load_runs(roots)
    from report_timeseries import load_observations

    observations = load_observations(roots, data)
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ReportDataError(
            "Matplotlib is required for reporting; install the optional .[reporting] extra"
        ) from exc
    sources = {str(p): _hash(p) for root in roots for p in root.rglob("*.json") if p.is_file()}
    output.mkdir(parents=True)
    rows = []
    for name in data["datasets"]:
        for key in PRIMARY:
            stat = data["summary"][name]["metrics"][key]
            rows.append(
                {
                    "dataset": name,
                    "metric": "f1" if key == "weighted_f1" else key,
                    "value_percent": 100 * stat["mean"],
                    "training_seed": data["seeds"][0],
                    "evaluation": "VAL development",
                    "synthetic": data["synthetic"],
                }
            )
    _write_csv(output / "metrics_summary.csv", rows)
    figures = _plots(data, output)
    _html(data, output, figures)
    for path, before in sources.items():
        if _hash(Path(path)) != before:
            raise ReportDataError("An input JSON changed while generating the report")
    artifacts = {
        p.relative_to(output).as_posix(): _hash(p) for p in output.rglob("*") if p.is_file()
    }
    manifest = {
        "report_schema": 3,
        "complete": True,
        "report_source_sha256": _hash(Path(__file__)),
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "private_local_report": True,
        "scope": data["scope"],
        "synthetic": data["synthetic"],
        "framework_version": data["framework_version"],
        "seeds": data["seeds"],
        "configuration": data["configuration"],
        "source_files": data["provenance"] + observations["provenance"],
        "recorded_execution_source_hashes": {
            run["run_id"]: run.get("source_hashes") for run in data["runs"]
        },
        "summary": data["summary"],
        "figures": figures,
        "figure_measurements": _plot_scope(data),
        "additional_measurements": observations["datasets"],
        "presentation": {
            "training_seed": data["seeds"][0],
            "metric_labels": {"f1": "support-weighted F1"},
            "confusion_datasets": [name for name in CONFUSION_DATASETS if name in data["datasets"]],
            "png_dpi": 600,
            "pdf_format": "vector",
            "figure_suptitles": False,
            "class_aliases": CLASS_ALIASES,
            "confusion_canvas_inches": [8, 7.4],
            "confusion_matrix_side_inches": 5.6,
        },
        "artifact_sha256": artifacts,
        "input_json_files_unchanged": len(sources),
        "selection": {
            run["run_id"]: {name: run["datasets"][name]["selection"] for name in data["datasets"]}
            for run in data["runs"]
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        "--runs",
        dest="run",
        required=True,
        type=Path,
        help="One completed v3 or v5 MPC experiment directory",
    )
    parser.add_argument("--out", type=Path, help="NEW report directory outside input experiments")
    args = parser.parse_args(argv)
    if args.out is None:
        from _iov_layout import project_root

        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        args.out = (
            project_root(__file__)
            / "outputs"
            / f"presentation_{stamp}_{uuid.uuid4().hex[:6]}"
            / "private_DO_NOT_UPLOAD"
        )
    try:
        report = generate_report([args.run], args.out)
    except (ReportDataError, OSError, ValueError) as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 2
    print(f'REPORT COMPLETE: {len(report["figures"])} figures; seed {report["seeds"][0]}')
    print(f'Report: {args.out.resolve() / "report.html"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
