"""005 result folders (contracts/results.md): OUTPUT_DIR/results/{kind}-{ts}-{hex}/, files written first,
manifest.json last (its presence marks the folder complete). Aggregates and opaque IDs only."""
import csv
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
COLUMNS = {
    "metrics.csv": ["system", "scope", "group", "label", "metric", "value", "ci_low", "ci_high", "support",
                    "n_segments", "status", "tag"],
    "confusion.csv": ["system", "human_primary", "model_primary", "count"],
    "signal_indicators.csv": ["indicator", "support", "tp", "fp", "fn", "precision", "recall", "f1", "ci_low",
                              "ci_high", "not_available", "status", "tag", "system"],
    "ablation.csv": ["system", "n_segments", "primary_accuracy", "primary_accuracy_ci_low", "primary_accuracy_ci_high",
                     "macro_f1_activity", "macro_f1_activity_ci_low", "macro_f1_activity_ci_high",
                     "macro_f1_category", "macro_f1_category_ci_low", "macro_f1_category_ci_high",
                     "macro_f1_actor", "macro_f1_actor_ci_low", "macro_f1_actor_ci_high", "kappa_primary",
                     "kappa_category", "rules_fired", "rules_not_applicable"],
    "topup.csv": ["label", "predicted_positive", "tp", "precision"],
    "failure_gallery.csv": ["rank", "segment_id", "video_id", "model_primary", "human_primary", "model_strength"],
    "learned_predictions.csv": ["segment_id", "video_id", "primary", "primary_prob", "activities", "categories",
                                "actors", "tag"],
    "indicators.csv": ["indicator", "scope_type", "scope", "value", "unit", "numerator", "denominator", "base", "tag",
                       "verified_share", "status"],
    "headline.csv": ["indicator", "phase", "value", "unit", "tag", "verified_share", "change_vs_early", "status"],
    "time_breakdown.csv": ["scope_type", "scope", "youth_s", "staff_led_verified_s", "staff_led_model_s", "idle_s",
                           "total_s"],
}
INSIGHT_TABLES = {"indicators.csv": "indicators", "headline.csv": "headline", "time_breakdown.csv": "time_breakdown"}
UNITS = {"share": lambda v: f"{100 * v:.0f}%", "minutes": lambda v: f"{v:.1f} min", "count": lambda v: f"{v:.0f}",
         "ratio": lambda v: f"{v:.2f}", "per_minute": lambda v: f"{v:.2f} per min", "index_0_100": lambda v: f"{v:.0f}"}


def show(value, unit) -> str:
    return "n/a" if value is None else UNITS.get(unit, lambda v: f"{v:.3g}")(value)


def show_change(value, unit) -> str:
    """A change between phases: percentage points for shares, signed otherwise."""
    if value is None:
        return ""
    if unit == "share":
        return f"{100 * value:+.0f} pp"
    return ("+" if value >= 0 else "-") + show(abs(value), unit)


def insights_summary(p: dict) -> str:
    lines = ["# Program indicators (youth only, program level)", "",
             f"Model fallback: {p['labels_source']} (tag: {p['model_tag']}). Staff-verified labels always come first.",
             ""]
    lines += ["## Headline: early program vs final showcase", "", "| indicator | phase | value | change vs early | tag |",
              "|---|---|---|---|---|"]
    for r in p["headline"]:
        if r["status"] != "ok":
            lines.append(f"| {r['indicator']} | {r['phase']} | {r['status']} |  | (no tag: {r['status']}) |")
            continue
        change = show_change(r["change_vs_early"], r["unit"])
        lines.append(f"| {r['indicator']} | {r['phase']} | {show(r['value'], r['unit'])} | {change} | {r['tag']} |")
    tb = next(b for b in p["time_breakdown"] if b["scope_type"] == "all")
    lines += ["", "## Where the time goes (all videos)", "",
              f"- Youth time: {tb['youth_s'] / 60:.1f} min (tag: {p['model_tag']} where not staff-verified)",
              f"- Staff-led time: {tb['staff_led_verified_s'] / 60:.1f} min staff-verified (tag: staff-verified), "
              f"{tb['staff_led_model_s'] / 60:.1f} min by the model rule (tag: {p['model_tag']}); the model rule "
              f"excluded {100 * p['model_staff_share']:.0f}% of all time",
              f"- Idle / setup time: {tb['idle_s'] / 60:.1f} min (tag: {UNVALIDATED_TAG})"]
    if p["excluded_videos"]:
        lines.append(f"- Left out of the headline (phase unknown): {', '.join(p['excluded_videos'])}")
    lines += ["", "Positive expression rate: observable, unvalidated; segments with faces only.", "",
              "## Limitations", "", *[f"- {x}" for x in p["limitations"]], ""]
    return "\n".join(lines)


UNVALIDATED_TAG = "model estimate (unvalidated)"
TABLES = {"metrics.csv": "metrics", "confusion.csv": "confusion", "signal_indicators.csv": "signal_indicators",
          "ablation.csv": "ablation", "topup.csv": "topup", "failure_gallery.csv": "failure_gallery",
          "learned_predictions.csv": "learned_predictions"}


def results_root(root=None) -> Path:
    from tpa_index.paths import output_dir

    return Path(root or output_dir()) / "results"


def new_result_dir(kind: str, root=None) -> Path:
    path = results_root(root) / f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_result(kind: str, root=None) -> Path | None:
    """Newest complete folder of this kind (has manifest.json); incomplete folders are ignored."""
    base = results_root(root)
    done = sorted(p for p in base.glob(f"{kind}-*") if (p / "manifest.json").exists()) if base.exists() else []
    return done[-1] if done else None


def _fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return round(v, 6) if isinstance(v, float) else v


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in columns})


def _json(obj) -> str:
    def default(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if hasattr(o, "item"):
            return o.item()
        return str(o)
    return json.dumps(obj, indent=2, default=default, sort_keys=False)


def _ci(value, ci, pct=True) -> str:
    if value is None:
        return "undefined"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    lo, hi = ci or (None, None)
    return f"{f(value)} (95% CI {f(lo)}–{f(hi)}" if lo is not None else f"{f(value)} (95% CI undefined"


def evaluate_summary(p: dict) -> str:
    es, lines = p["eval_set"], ["# Evaluation: model vs blind labels (eval videos only)", ""]
    per_video = ", ".join(f"{v}: {n}" for v, n in sorted(es["per_video"].items()))
    lines += [f"Labeled eval segments: {es['labeled_eval_segments']} ({per_video}); skipped {es['skipped']}, "
              f"not scored {es['not_scored']}, dev labels ignored {es['dev_labels_ignored']}.", ""]
    for h in p["headline"]:
        n = h["n_segments"]
        lines += [f"## {h['system']}", "",
                  f"- Primary activity accuracy: {_ci(h['primary_accuracy'], h['primary_accuracy_ci'])}, n={n})",
                  *[f"- Macro-F1 {g}: {_ci(h[f'macro_f1_{g}'], h[f'macro_f1_{g}_ci'])}, n={n})"
                    for g in ("activity", "category", "actor")],
                  f"- Kappa, primary activity: {_ci(h['kappa_primary'], h['kappa_primary_ci'], pct=False)}, n={n})."
                  f" {h['kappa_primary_sentence']}",
                  f"- Kappa, category: {_ci(h['kappa_category'], h['kappa_category_ci'], pct=False)}, "
                  f"n={n - h['category_kappa_excluded']}). {h['kappa_category_sentence']}", ""]
    lines += ["## Ablation (same segments for every system)", "",
              "| system | primary accuracy | macro-F1 activity | kappa, primary |", "|---|---|---|---|"]
    for h in p["headline"]:
        n = h["n_segments"]
        lines.append(f"| {h['system']} | {_ci(h['primary_accuracy'], h['primary_accuracy_ci'])}, n={n}) | "
                     f"{_ci(h['macro_f1_activity'], h['macro_f1_activity_ci'])}, n={n}) | "
                     f"{_ci(h['kappa_primary'], h['kappa_primary_ci'], pct=False)}, n={n}) |")
    lines += ["", "## Signal-based indicators", "",
              "| system | indicator | support | precision | recall | F1 (95% CI) | not available |",
              "|---|---|---|---|---|---|---|"]
    for r in p["signal_indicators"]:
        n_avail = p["eval_set"]["labeled_eval_segments"] - r["not_available"]
        f1 = "too few to score" if r["status"] != "ok" else \
            _ci(r["f1"], (r["ci_low"], r["ci_high"])) + f", n={n_avail})"
        prec = "" if r["precision"] is None else f"{100 * r['precision']:.0f}%"
        rec = "" if r["recall"] is None else f"{100 * r['recall']:.0f}%"
        lines.append(f"| {r['system']} | {r['indicator']} | {r['support']} | {prec} | {rec} | {f1} | "
                     f"{r['not_available']} |")
    lines += ["", "## Limitations", "", *[f"- {x}" for x in p["limitations"]], ""]
    return "\n".join(lines)


# ---- charts (white background, TPA palette, tag in every title or footnote) ----------------------------------------
PALETTE = ["#7851B9", "#0099B3", "#FF4255"]
TEXT, GREY = "#201C25", "#C9C4CF"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                         "axes.edgecolor": TEXT, "axes.titlecolor": TEXT, "font.size": 10})
    return plt


def _save(fig, out: Path, name: str, footnote: str) -> str:
    import textwrap

    width_chars = int(fig.get_figwidth() * 16)
    lines = textwrap.wrap(footnote, width_chars) or [""]
    height_in = fig.get_figheight()
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, color=TEXT, ha="left", va="bottom")
    fig.tight_layout(rect=(0, min(0.45, (0.18 * len(lines) + 0.15) / height_in), 1, 1))
    fig.savefig(out / name, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return name


def _err(values, cis):
    lo = [v - c[0] if v is not None and c[0] is not None else 0 for v, c in zip(values, cis)]
    hi = [c[1] - v if v is not None and c[1] is not None else 0 for v, c in zip(values, cis)]
    return [lo, hi]


def evaluate_charts(p: dict, out: Path) -> list[str]:
    import numpy as np

    plt, files = _plt(), []
    heads, n = p["headline"], p["eval_set"]["labeled_eval_segments"]
    note = f"Tag: metric (model vs blind labels, eval videos only) · error bars: 95% block-bootstrap CI · n={n}"
    metrics = [("primary_accuracy", "Primary accuracy"), ("macro_f1_activity", "Macro-F1 activity"),
               ("macro_f1_category", "Macro-F1 category"), ("macro_f1_actor", "Macro-F1 actor")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, w = np.arange(len(metrics)), 0.8 / max(len(heads), 1)
    for i, h in enumerate(heads):
        vals = [h[k] if h[k] is not None else 0 for k, _ in metrics]
        cis = [h[f"{k}_ci"] for k, _ in metrics]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=PALETTE[i % 3], label=h["system"],
               yerr=_err(vals, cis), capsize=3, ecolor=TEXT)
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Ablation: zero-shot vs signal rules vs learned (metric · 95% CI)")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8)
    files.append(_save(fig, out, "ablation.png", note))

    rows = [r for r in p["metrics"] if r["group"] == "activity" and r["metric"] == "f1" and r["scope"] == "all"]
    labels_ok = sorted({r["label"] for r in rows if r["status"] == "ok"})
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(labels_ok) + 1.5)))
    y = np.arange(len(labels_ok))
    for i, h in enumerate(heads):
        by = {r["label"]: r["value"] or 0 for r in rows if r["system"] == h["system"]}
        ax.barh(y + i * w - 0.4 + w / 2, [by.get(lab, 0) for lab in labels_ok], w, color=PALETTE[i % 3],
                label=h["system"])
    ax.set_yticks(y, labels_ok)
    ax.set_xlim(0, 1)
    ax.set_title("Per-activity F1, labels with support ≥ 5 (metric)")
    ax.legend(frameon=False, loc="lower right")
    files.append(_save(fig, out, "per_label_f1.png", note))

    from matplotlib.colors import LinearSegmentedColormap

    conf = [c for c in p["confusion"] if c["system"] == heads[0]["system"]]
    classes = sorted({c["human_primary"] for c in conf} | {c["model_primary"] for c in conf})
    grid = np.zeros((len(classes), len(classes)))
    for c in conf:
        grid[classes.index(c["human_primary"]), classes.index(c["model_primary"])] = c["count"]
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * len(classes), 1.2 + 0.55 * len(classes)))
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("tpa", ["#FFFFFF", PALETTE[0]]))
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("model primary")
    ax.set_ylabel("human primary")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if grid[i, j]:
                ax.text(j, i, int(grid[i, j]), ha="center", va="center",
                        color="white" if grid[i, j] > grid.max() / 2 else TEXT)
    ax.set_title(f"Primary activity confusion, {heads[0]['system']} (metric)")
    files.append(_save(fig, out, "confusion.png", f"Tag: metric · counts of labeled eval segments · n={n}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(heads))
    conf_acc = [h["uncertainty"]["accuracy_confident"] or 0 for h in heads]
    flag_acc = [h["uncertainty"]["accuracy_flagged"] or 0 for h in heads]
    ax.bar(x - 0.2, conf_acc, 0.4, color=PALETTE[0], label="confident (unflagged)")
    ax.bar(x + 0.2, flag_acc, 0.4, color=PALETTE[2], label="flagged")
    for i, h in enumerate(heads):
        ax.text(i, 1.02, f"coverage {100 * (h['uncertainty']['coverage'] or 0):.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, [h["system"] for h in heads])
    ax.set_ylim(0, 1.12)
    ax.set_title("Primary accuracy on confident vs flagged segments (metric)")
    ax.legend(frameon=False, loc="lower right")
    files.append(_save(fig, out, "coverage.png", note))
    return files


def insights_charts(p: dict, out: Path) -> list[str]:
    plt, files = _plt(), []
    head = [r for r in p["headline"] if r["status"] == "ok"]
    inds = list(dict.fromkeys(r["indicator"] for r in p["headline"]))
    fig, axes = plt.subplots(1, len(inds), figsize=(3.2 * len(inds), 4.8))
    notes = []
    for ax, ind in zip(axes if len(inds) > 1 else [axes], inds):
        rs = [r for r in head if r["indicator"] == ind]
        ax.bar([r["phase"] for r in rs], [r["value"] for r in rs], color=PALETTE[:len(rs)])
        for i, r in enumerate(rs):
            ax.text(i, r["value"], show(r["value"], r["unit"]), ha="center", va="bottom", fontsize=8)
            notes.append(f"{ind.replace('_', ' ')}, {r['phase']}: {r['tag']}")
        ax.set_title(ind.replace("_", " "), fontsize=10)
    fig.suptitle("Early program vs final showcase (youth only; tags below)")
    files.append(_save(fig, out, "headline.png", "Tags: " + "; ".join(notes)))

    tb = [b for b in p["time_breakdown"] if b["scope_type"] == "phase"]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [b["scope"] for b in tb]
    bottom = [0.0] * len(tb)
    for key, color, label in (("youth_s", PALETTE[0], "youth"),
                              ("staff_led_verified_s", PALETTE[1], "staff-led (staff-verified)"),
                              ("staff_led_model_s", PALETTE[2], f"staff-led ({p['model_tag']})"),
                              ("idle_s", GREY, "idle / setup")):
        vals = [b[key] / 60 for b in tb]
        ax.bar(names, vals, bottom=bottom, color=color, label=label)
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.set_ylabel("minutes")
    ax.set_title("Where the time goes, per phase (mixed: staff-verified + model)")
    ax.legend(frameon=False, fontsize=8)
    files.append(_save(fig, out, "time_breakdown.png",
                       f"Tags: staff-led split into staff-verified and {p['model_tag']}; idle from the model's idle rule"))
    return files


def write_result(kind: str, payload: dict, root=None) -> Path:
    """Write every file of the result folder, then manifest.json last (atomic)."""
    from tpa_index.runs import git_commit

    out = new_result_dir(kind, root)
    files = []
    if kind == "evaluate":
        for name, key in TABLES.items():
            _write_csv(out / name, COLUMNS[name], payload[key])
            files.append(name)
        (out / "metrics.json").write_text(_json({k: payload[k] for k in (
            "metrics", "headline", "eval_set", "limitations", "guideline_versions")}))
        (out / "summary.md").write_text(evaluate_summary(payload))
        files += ["metrics.json", "summary.md"]
    elif kind == "insights":
        for name, key in INSIGHT_TABLES.items():
            _write_csv(out / name, COLUMNS[name], payload[key])
            files.append(name)
        (out / "indicators.json").write_text(_json({k: payload[k] for k in (
            "indicators", "headline", "time_breakdown", "excluded_videos", "labels_source", "model_tag",
            "phase_source", "model_staff_share", "limitations")}))
        (out / "summary.md").write_text(insights_summary(payload))
        files += ["indicators.json", "summary.md"]
    else:
        raise ValueError(f"unknown result kind {kind!r}")
    files += evaluate_charts(payload, out) if kind == "evaluate" else insights_charts(payload, out)
    manifest = {"schema_version": SCHEMA_VERSION, "kind": kind,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git_commit(),
                "config_hash": payload["config_hash"], "guideline_versions": payload.get("guideline_versions", []),
                "official_runs": payload["official_runs"],
                **({"phase_source": payload["phase_source"]} if "phase_source" in payload else {}),
                "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files},
                "limitations": payload["limitations"]}
    tmp = out / "manifest.json.tmp"
    tmp.write_text(_json(manifest))
    os.replace(tmp, out / "manifest.json")
    return out
