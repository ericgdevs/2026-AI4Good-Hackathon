"""The six output tables: pyarrow schemas (source of truth), validation on write, Parquet/DuckDB I/O."""
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tpa_index.config import LABEL_GROUPS
from tpa_index.paths import output_dir, repo_root

_s, _f64, _f32, _i32, _b = pa.string(), pa.float64(), pa.float32(), pa.int32(), pa.bool_()
_ts = pa.timestamp("us", tz="UTC")
_list = pa.list_(pa.string())


def _schema(*fields):
    return pa.schema([pa.field(name, typ, nullable=nullable) for name, typ, nullable in fields])


SCHEMAS: dict[str, pa.Schema] = {
    "videos": _schema(
        ("video_id", _s, False), ("filename", _s, False), ("duration_s", _f64, False), ("fps", _f64, False),
        ("orientation", _s, False), ("kind", _s, False), ("phase", _s, False), ("cohort", _s, True),
        ("split", _s, False),
    ),
    "segments": _schema(
        ("segment_id", _s, False), ("video_id", _s, False), ("start_s", _f64, False), ("end_s", _f64, False),
        ("person_count", _i32, True), ("faces_detected", _i32, True), ("has_speech", _b, True),
        ("transcript_chars", _i32, True), ("is_idle", _b, True), ("idle_score", _f32, True),
        ("visual_change", _f32, True),
    ),
    "predictions": _schema(
        ("segment_id", _s, False), ("label", _s, False), ("label_group", _s, False), ("score", _f32, False),
        ("model", _s, False), ("run_id", _s, False),
    ),
    "moments": _schema(
        ("moment_id", _s, False), ("video_id", _s, False), ("start_s", _f64, False), ("end_s", _f64, False),
        ("primary_label", _s, False), ("secondary_labels", _list, False), ("indicators", _list, False),
        ("confidence", _f32, False), ("needs_review", _b, False), ("run_id", _s, False),
    ),
    "runs": _schema(
        ("run_id", _s, False), ("stage", _s, False), ("model", _s, True), ("config_hash", _s, False),
        ("hardware", _s, False), ("n_segments", _i32, False), ("runtime_s", _f64, False),
        ("peak_mem_gb", _f64, True), ("started_at", _ts, False), ("git_commit", _s, True),
    ),
    "reviews": _schema(
        ("review_id", _s, False), ("target_type", _s, False), ("target_id", _s, False), ("label", _s, False),
        ("label_role", _s, False), ("action", _s, False), ("new_label", _s, True), ("source", _s, False),
        ("mode", _s, False), ("guideline_version", _s, False), ("reviewer", _s, False), ("note", _s, True),
        ("created_at", _ts, False),
    ),
    "blind_queue": _schema(
        ("queue", _s, False), ("position", _i32, False), ("segment_id", _s, False), ("video_id", _s, False),
        ("start_s", _f64, False), ("end_s", _f64, False), ("target_label", _s, True), ("seed", pa.int64(), False),
        ("created_at", _ts, False),
    ),
}

# review-row roles (a 001 contract enum; "observation" added in 003)
LABEL_ROLES = ("primary", "secondary", "indicator", "observation")

RULES: dict[str, dict] = {
    "videos": {
        "pk": ("video_id",),
        "enums": {"split": {"dev", "eval"}, "orientation": {"landscape", "portrait"},
                  "kind": {"live", "edited", "clip", "promo"}, "phase": None},  # phase: from pipeline.yaml
        "positive": ("duration_s", "fps"),
    },
    "segments": {
        "pk": ("segment_id",),
        "nonneg": ("start_s", "person_count", "faces_detected", "transcript_chars"),
        "unit": ("idle_score", "visual_change"),
        "times": True,
    },
    "predictions": {
        "pk": ("segment_id", "label", "model", "run_id"),
        "enums": {"label_group": set(LABEL_GROUPS)},
        "unit": ("score",),
    },
    "moments": {"pk": ("moment_id",), "unit": ("confidence",), "nonneg": ("start_s",), "times": True},
    "runs": {"pk": ("run_id",), "positive": ("runtime_s",), "nonneg": ("n_segments", "peak_mem_gb")},
    "blind_queue": {"pk": ("queue", "position"), "enums": {"queue": {"main", "overlap", "topup"}},
                    "nonneg": ("position", "start_s"), "times": True},
    "reviews": {
        "pk": ("review_id",),
        "enums": {"target_type": {"segment", "moment"}, "label_role": set(LABEL_ROLES),
                  "action": {"accept", "edit", "reject", "defer", "add"}, "source": {"ai", "staff"},
                  "mode": {"blind", "review"}},
    },
}


class SchemaError(Exception):
    def __init__(self, table: str, column: str, reason: str):
        super().__init__(f"{table}.{column}: {reason}")
        self.table, self.column, self.reason = table, column, reason


def validate(table: str, data, phases=None) -> pa.Table:
    """Return data as a pa.Table matching SCHEMAS[table], or raise SchemaError."""
    if table not in SCHEMAS:
        raise SchemaError(table, "*", "unknown table")
    schema, rules = SCHEMAS[table], RULES[table]
    names = schema.names

    # 1. exact column set (this is also what keeps identity columns out)
    if isinstance(data, pa.Table):
        cols = list(data.column_names)
        rows = None
    else:
        rows = list(data)
        cols = list(dict.fromkeys(k for r in rows for k in r)) if rows else names
        for r in rows:
            for name in names:
                if name not in r:
                    raise SchemaError(table, name, "missing column")
    for name in names:
        if name not in cols:
            raise SchemaError(table, name, "missing column")
    for name in cols:
        if name not in names:
            raise SchemaError(table, name, "unexpected column (not in contract)")

    # 2. types, 3. nulls
    arrays = []
    for field in schema:
        try:
            if rows is None:
                arr = data.column(field.name).cast(field.type)
            else:
                arr = pa.array([r[field.name] for r in rows], type=field.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError) as e:
            raise SchemaError(table, field.name, f"wrong type ({type(e).__name__})") from None
        if not field.nullable and arr.null_count:
            raise SchemaError(table, field.name, "null in non-nullable column")
        arrays.append(arr)
    tbl = pa.Table.from_arrays(arrays, schema=schema)

    # 4. enums
    for col, allowed in rules.get("enums", {}).items():
        if allowed is None:
            if phases is None:
                from tpa_index.config import load_pipeline
                phases = load_pipeline()["video_phases"]
            allowed = set(phases)
        bad = {v for v in tbl[col].to_pylist() if v is not None and v not in allowed}
        if bad:
            raise SchemaError(table, col, f"value not allowed: {sorted(bad)[0]!r}")

    # 5. ranges
    for kind, test, reason in (("unit", lambda v: 0 <= v <= 1, "must be within 0-1"),
                               ("positive", lambda v: v > 0, "must be > 0"),
                               ("nonneg", lambda v: v >= 0, "must be >= 0")):
        for col in rules.get(kind, ()):
            if any(v is not None and not test(v) for v in tbl[col].to_pylist()):
                raise SchemaError(table, col, reason)

    # 6. times
    if rules.get("times"):
        if any(e <= s for s, e in zip(tbl["start_s"].to_pylist(), tbl["end_s"].to_pylist())):
            raise SchemaError(table, "end_s", "must be greater than start_s")

    # 7. primary key, review new_label rule
    pk = rules["pk"]
    keys = list(zip(*(tbl[c].to_pylist() for c in pk)))
    if len(set(keys)) != len(keys):
        raise SchemaError(table, ",".join(pk), "duplicate primary key")
    if table == "reviews":
        for action, new_label in zip(tbl["action"].to_pylist(), tbl["new_label"].to_pylist()):
            if (action in ("edit", "add")) != (new_label is not None):
                raise SchemaError(table, "new_label", "required for edit/add and must be empty otherwise")
    return tbl


_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _runs_root(root) -> Path:
    return Path(root or output_dir()) / "runs"


def write_table(table: str, data, run_id: str, root=None) -> Path:
    """Validate, then write {root}/runs/{run_id}/{table}.parquet. Never overwrites."""
    if not _RUN_ID.match(run_id):
        raise SchemaError(table, "run_id", "run_id may contain only letters, digits, '_', '.', '-'")
    tbl = validate(table, data)
    path = _runs_root(root) / run_id / f"{table}.parquet"
    if path.exists():
        raise FileExistsError(f"{table} already written for run {run_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)
    return path


def read_table(table: str, root=None, run_ids=None) -> pa.Table:
    """Complete runs (folders that contain runs.parquet) concatenated; run_ids limits which run folders are read."""
    runs = _runs_root(root)
    keep = None if run_ids is None else set(run_ids)
    files = [d / f"{table}.parquet" for d in sorted(runs.glob("*"))
             if (keep is None or d.name in keep)
             and (d / "runs.parquet").exists() and (d / f"{table}.parquet").exists()] if runs.exists() else []
    if not files:
        return SCHEMAS[table].empty_table()
    return pa.concat_tables([_conform(pq.read_table(f), SCHEMAS[table]) for f in files])


def _conform(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast to the current schema; nullable columns added after a file was written read back as null."""
    for field in schema:
        if field.name not in tbl.column_names and field.nullable:
            tbl = tbl.append_column(field.name, pa.nulls(tbl.num_rows, field.type))
    return tbl.select(schema.names).cast(schema)


_reviews_lock = threading.Lock()


def _reviews_db(root) -> Path:
    return Path(root or output_dir()) / "reviews.duckdb"


def _open_reviews(root) -> duckdb.DuckDBPyConnection:
    """Read-write connection with both tables created if missing (idempotent), so a fresh database is complete."""
    path = _reviews_db(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    for name in ("reviews", "blind_queue"):
        con.register("empty", SCHEMAS[name].empty_table())
        con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM empty")
        con.unregister("empty")
    return con


def _read_reviews_table(root, name: str, order: str) -> pa.Table:
    """A table from reviews.duckdb, or an empty one with its schema if the file or table is missing."""
    path = _reviews_db(root)
    if not path.exists():
        return SCHEMAS[name].empty_table()
    # same settings as writes: DuckDB refuses a second in-process connection with different ones
    con = duckdb.connect(str(path))
    try:
        # databases written before _open_reviews may lack a table
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchall()[0][0]:
            return SCHEMAS[name].empty_table()
        return con.sql(f"SELECT * FROM {name} ORDER BY {order}").to_arrow_table().cast(SCHEMAS[name])
    finally:
        con.close()


def append_reviews(rows, root=None) -> int:
    """Append review entries. The log is append-only: there is no update or delete."""
    rows = rows.to_pylist() if isinstance(rows, pa.Table) else [dict(r) for r in rows]
    for r in rows:
        r.setdefault("review_id", uuid.uuid4().hex)
        r.setdefault("created_at", datetime.now(timezone.utc))
    tbl = validate("reviews", rows)
    with _reviews_lock:
        con = _open_reviews(root)
        try:
            con.register("incoming", tbl)
            con.execute("INSERT INTO reviews SELECT * FROM incoming")
        finally:
            con.close()
    return tbl.num_rows


def append_blind_queue(rows, root=None) -> int:
    """Write one blind queue, once. Queues are never modified, so a stable sample survives later runs."""
    tbl = validate("blind_queue", list(rows))
    queues = set(tbl["queue"].to_pylist())
    if len(queues) != 1:
        raise SchemaError("blind_queue", "queue", "write exactly one queue at a time")
    ids = tbl["segment_id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise SchemaError("blind_queue", "segment_id", "duplicate segment in queue")
    (queue,) = queues
    with _reviews_lock:
        con = _open_reviews(root)
        try:
            con.register("incoming", tbl)
            if con.execute("SELECT count(*) FROM blind_queue WHERE queue = ?", [queue]).fetchall()[0][0]:
                raise SchemaError("blind_queue", "queue", f"queue {queue} already exists (queues are written once)")
            con.execute("INSERT INTO blind_queue SELECT * FROM incoming")
        finally:
            con.close()
    return tbl.num_rows


def read_blind_queue(queue=None, root=None) -> pa.Table:
    tbl = _read_reviews_table(root, "blind_queue", "queue, position")
    if queue:
        import pyarrow.compute as pc

        tbl = tbl.filter(pc.equal(tbl["queue"], queue))
    return tbl


def read_reviews(root=None) -> pa.Table:
    return _read_reviews_table(root, "reviews", "created_at, review_id")


def validate_dataset(root=None) -> None:
    """Cross-table reference checks over all complete runs."""
    t = {name: read_table(name, root) for name in ("videos", "segments", "predictions", "moments", "runs")}
    ids = {
        "video_id": set(t["videos"]["video_id"].to_pylist()),
        "segment_id": set(t["segments"]["segment_id"].to_pylist()),
        "run_id": set(t["runs"]["run_id"].to_pylist()),
    }
    for table, col in (("segments", "video_id"), ("moments", "video_id"), ("predictions", "segment_id"),
                       ("predictions", "run_id"), ("moments", "run_id")):
        missing = set(t[table][col].to_pylist()) - ids[col]
        if missing:
            raise SchemaError(table, col, f"{len(missing)} value(s) not found in the referenced table")


# ---- official runs (002) ----------------------------------------------------------------------------------------
OFFICIAL_RUNS = Path(os.environ.get("TPA_OFFICIAL_RUNS", repo_root() / "config/official_runs.yaml"))


def load_official(path=None) -> dict[str, list[str]]:
    """{stage: [run_id, ...]} from config/official_runs.yaml (run IDs only)."""
    import yaml

    path = Path(path or OFFICIAL_RUNS)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    for stage, ids in data.items():
        if ids is not None and not isinstance(ids, list):
            raise SchemaError("official_runs", stage, "must be a list of run IDs")
    return {stage: list(ids or []) for stage, ids in data.items()}


def _run_videos(run_id: str, root) -> list[str]:
    path = _runs_root(root) / run_id / "videos.parquet"
    return pq.read_table(path, columns=["video_id"])["video_id"].to_pylist() if path.exists() else []


def _official_by_video(official: dict, stage: str, root) -> dict[str, str]:
    """video_id -> run_id for one stage; every listed run must be complete and videos must not repeat."""
    by_video: dict[str, str] = {}
    for run_id in official.get(stage, []):
        if not (_runs_root(root) / run_id / "runs.parquet").exists():
            raise SchemaError("official_runs", stage, f"run {run_id} not found or incomplete")
        for vid in _run_videos(run_id, root):
            if vid in by_video:
                raise SchemaError("official_runs", stage,
                                  f"runs {by_video[vid]} and {run_id} both cover video {vid}")
            by_video[vid] = run_id
    return by_video


def _video_of(segment_id: str) -> str:
    return segment_id.rsplit("_", 1)[0]


def read_official(table: str, stage: str | None = None, root=None) -> pa.Table:
    """Rows from the official runs only. Result-facing code must use this, not read_table.
    Default: moments and segments come from an official moments run for a video when listed, else predict;
    predictions take the moments run's rules rows in place of predict's."""
    official = load_official()
    if stage is not None:
        return read_table(table, root, run_ids=sorted(set(_official_by_video(official, stage, root).values())))
    predict = _official_by_video(official, "predict", root)
    rebuilt = _official_by_video(official, "moments", root)
    if table in ("moments", "segments"):
        chosen = {**predict, **rebuilt}
        return read_table(table, root, run_ids=sorted(set(chosen.values())))
    base = read_table(table, root, run_ids=sorted(set(predict.values())))
    if table != "predictions" or not rebuilt:
        return base
    import pyarrow.compute as pc

    replaced = set(rebuilt)
    keep = [not (m == "rules" and _video_of(s) in replaced)
            for m, s in zip(base["model"].to_pylist(), base["segment_id"].to_pylist())]
    rules = read_table("predictions", root, run_ids=sorted(set(rebuilt.values())))
    rules = rules.filter(pc.equal(rules["model"], "rules"))
    return pa.concat_tables([base.filter(pa.array(keep, pa.bool_())), rules])


def list_runs(root=None) -> list[dict]:
    """One dict per complete run: ids, stage, video_ids, hardware, counts, runtime, peak memory (no paths)."""
    runs = _runs_root(root)
    out = []
    for d in sorted(runs.glob("*")) if runs.exists() else []:
        if not (d / "runs.parquet").exists():
            continue
        preds = d / "predictions.parquet"
        for r in _conform(pq.read_table(d / "runs.parquet"), SCHEMAS["runs"]).to_pylist():
            out.append({"run_id": r["run_id"], "stage": r["stage"], "video_ids": ",".join(_run_videos(d.name, root)),
                        "hardware": r["hardware"], "n_segments": r["n_segments"],
                        "n_predictions": pq.ParquetFile(preds).metadata.num_rows if preds.exists() else 0,
                        "runtime_s": r["runtime_s"], "peak_mem_gb": r["peak_mem_gb"]})
    return out
