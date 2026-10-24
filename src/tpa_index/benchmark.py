"""Mac benchmark (006 US5): run every pipeline stage for one video on this machine, then compare with the official
HPC runs per hour of video. Each stage already writes its runs row (runtime = processing only, because commands load
their models before track_run starts), so model loading = the step's wall time minus the runtime it recorded."""
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("segment", "predict", "transcript", "audio", "people", "expression", "moments")
SIGNAL_STAGES = ("transcript", "audio", "people", "expression")
NOTES = (
    "Processing time excludes model loading; loading is shown once per stage.",
    "The Mac uses the smaller Mac models from pipeline.yaml; HPC numbers are the official runs.",
    "Benchmark runs are not official runs and change no result.",
)


def steps(video_id: str, videos: str | None = None) -> list[tuple[str, list[str]]]:
    """(stage, argv) in run order; each signal stage is its own call so its model load is timed on its own."""
    vids = ["--videos", videos] if videos else []
    out = [("segment", ["segment", "--video-id", video_id, *vids]),
           ("predict", ["predict", "--video-id", video_id, *vids])]
    out += [(s, ["signals", "--stages", s, "--video-id", video_id, *vids, "--force"]) for s in SIGNAL_STAGES]
    return out + [("moments", ["moments", "--video-id", video_id])]


def run_benchmark(video_id: str, videos: str | None = None, runner=None, clock=time.perf_counter,
                  root=None) -> list[dict]:
    """Run each step in order; a failing step is recorded and the next ones still run. Never edits official runs."""
    from tpa_index.paths import redact
    from tpa_index.schema import list_runs

    if runner is None:
        from tpa_index.cli import main as runner
    results = []
    for stage, argv in steps(video_id, videos):
        before = {r["run_id"] for r in list_runs(root)}
        t0 = clock()
        error = None
        try:
            code = runner(argv)
            if code:
                error = f"exit code {code}"
        except Exception as e:  # noqa: BLE001 - one stage failing must not stop the others
            # the summary goes into slides: redact data paths, then drop any other absolute path
            error = re.sub(r"/[^\s:'\"]+", "<path>", redact(f"{type(e).__name__}: {e}"))
        wall = clock() - t0
        new = [r["run_id"] for r in list_runs(root) if r["run_id"] not in before and r["stage"] == stage
               and video_id in r["video_ids"].split(",")]
        results.append({"stage": stage, "wall_s": wall, "run_ids": new, "error": error})
    return results


def _run_rows(run_ids: list[str], root) -> list[dict]:
    """runs rows plus the video-hours each run covers (from the run folder's videos table)."""
    from tpa_index.schema import read_table

    out = []
    for rid in run_ids:
        run = read_table("runs", root, run_ids=[rid]).to_pylist()
        videos = read_table("videos", root, run_ids=[rid]).to_pylist()
        if run:
            out.append({**run[0], "video_hours": sum(v["duration_s"] for v in videos) / 3600})
    return out


def _per_hour(rows: list[dict]) -> float | None:
    hours = sum(r["video_hours"] for r in rows)
    return sum(r["runtime_s"] for r in rows) / hours if hours else None


def _peak(rows: list[dict]) -> float | None:
    vals = [r["peak_mem_gb"] for r in rows if r["peak_mem_gb"] is not None]
    return max(vals) if vals else None


def _names(rows: list[dict], key: str) -> str:
    return ", ".join(sorted({str(r[key]) for r in rows if r[key]})) or "-"


def summarize(results: list[dict], root=None) -> dict:
    from tpa_index.schema import load_official

    official = load_official()
    rows = []
    for res in results:
        mac = _run_rows(res["run_ids"], root)
        hpc = _run_rows(official.get(res["stage"], []), root)
        processing = sum(r["runtime_s"] for r in mac)
        status = (f"failed: {res['error']}" if res["error"] or not mac else
                  "no HPC run" if not hpc else "ok")
        rows.append({
            "stage": res["stage"],
            "mac_processing_s": processing if mac else None,
            "mac_load_s": max(res["wall_s"] - processing, 0.0) if mac else None,
            "mac_s_per_video_hour": _per_hour(mac),
            "mac_peak_mem_gb": _peak(mac),
            "mac_model": _names(mac, "model"), "mac_hardware": _names(mac, "hardware"),
            "hpc_s_per_video_hour": _per_hour(hpc),
            "hpc_peak_mem_gb": _peak(hpc),
            "hpc_model": _names(hpc, "model"), "hpc_hardware": _names(hpc, "hardware"),
            "status": status,
        })
    missing = [r["stage"] for r in rows if r["status"] != "ok"]

    def total(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) if vals else None

    rows.append({
        "stage": "total",
        "mac_processing_s": total("mac_processing_s"), "mac_load_s": total("mac_load_s"),
        "mac_s_per_video_hour": total("mac_s_per_video_hour"),
        "mac_peak_mem_gb": max((r["mac_peak_mem_gb"] for r in rows if r["mac_peak_mem_gb"] is not None), default=None),
        "mac_model": "-", "mac_hardware": _names(rows, "mac_hardware"),
        "hpc_s_per_video_hour": total("hpc_s_per_video_hour"),
        "hpc_peak_mem_gb": max((r["hpc_peak_mem_gb"] for r in rows if r["hpc_peak_mem_gb"] is not None), default=None),
        "hpc_model": "-", "hpc_hardware": _names(rows, "hpc_hardware"),
        "status": f"incomplete: missing {', '.join(missing)}" if missing else "ok",
    })
    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows,
            "notes": list(NOTES)}


def _fmt(v, digits=1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def markdown(summary: dict) -> str:
    head = ("| stage | Mac s / video-hour (processing) | Mac load s | Mac peak GB | HPC s / video-hour | HPC peak GB "
            "| models (Mac / HPC) | status |")
    lines = [head, "|" + "---|" * 8]
    for r in summary["rows"]:
        lines.append(f"| {r['stage']} | {_fmt(r['mac_s_per_video_hour'])} | {_fmt(r['mac_load_s'])} "
                     f"| {_fmt(r['mac_peak_mem_gb'], 2)} | {_fmt(r['hpc_s_per_video_hour'])} "
                     f"| {_fmt(r['hpc_peak_mem_gb'], 2)} | {r['mac_model']} / {r['hpc_model']} | {r['status']} |")
    return "\n".join(["# Mac vs HPC benchmark", "", *lines, "", *[f"- {n}" for n in summary["notes"]], ""])


def write_summary(summary: dict, root=None) -> tuple[Path, Path]:
    """{OUTPUT_DIR}/benchmarks/benchmark-{time}-{hex}.json and .md, never overwritten. No paths in the content."""
    from tpa_index.paths import output_dir

    out = Path(root or output_dir()) / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    while True:
        base = out / f"benchmark-{stamp}-{secrets.token_hex(2)}"
        try:
            with open(f"{base}.json", "x", encoding="utf-8") as f:
                f.write(json.dumps(summary, indent=2) + "\n")
            break
        except FileExistsError:
            continue
    with open(f"{base}.md", "x", encoding="utf-8") as f:
        f.write(markdown(summary))
    return Path(f"{base}.json"), Path(f"{base}.md")
