import copy
import json

import pytest

from tpa_index import schema
from tpa_index.clips import ClipCutter, session_dir
from tpa_index.mock import generate, write_clips


@pytest.fixture
def mock_env(tmp_path, monkeypatch, pipeline):
    """Mock tables + synthetic videos + an official runs file; small queues for speed."""
    run_id = generate(seed=0, n_videos=4)
    write_clips([v for v in schema.read_table("videos", run_ids=[run_id]).to_pylist() if v["split"] == "eval"])
    path = tmp_path / "official.yaml"
    path.write_text(f"predict: [{run_id}]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    cutter = ClipCutter(session_dir(), clip_height=240)
    small = copy.deepcopy(pipeline)
    small["app"].update(sample_size=12, overlap_size=4)
    yield {"cutter": cutter, "pipeline": small, "official": path}
    cutter.cleanup()


def _components(demo):
    return demo.config["components"]




def test_app_builds_and_blind_tab_shows_only_config_labels(mock_env, labels):
    from tpa_index.app import build_app
    from tpa_index.review import checklist

    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
    cl = checklist(labels, mock_env["pipeline"])
    by_id = {c["props"].get("elem_id"): c["props"] for c in _components(demo)}
    assert [v for _, v in by_id["blind-primary"]["choices"]] == cl["primary"]
    for i, (_, names) in enumerate(cl["groups"]):
        assert [v for _, v in by_id[f"blind-group-{i}"]["choices"]] == names
    blind_tab = [c for c in _components(demo)
                 if not str(c["props"].get("elem_id") or "").startswith(("review-", "search-", "scorecard-", "export-"))]
    page = json.dumps(blind_tab, default=str)
    model_names = [m.rsplit("/", 1)[-1] for m in mock_env["pipeline"]["models"]["main"].values()]
    for word in ("AI suggestion", "score", "confidence", *model_names):
        assert word not in page, word
    assert by_id["reviewer"]["placeholder"] == "staff first name or role"


def test_blind_flow_never_reads_model_output(mock_env, labels, monkeypatch):
    from tpa_index.app import BlindController, build_app
    from tpa_index.review import checklist

    real = schema.read_official

    def guarded(table, *a, **k):
        if table in ("predictions", "moments"):
            raise AssertionError(f"blind mode read {table}")
        return real(table, *a, **k)

    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
    monkeypatch.setattr(schema, "read_official", guarded)  # after build: the review tab loads moments at startup
    b = demo.blind
    cl = checklist(labels, mock_env["pipeline"])
    groups0 = [[] for _ in cl["groups"]]
    out = b.start("tester-a", "Main sample")
    state = out[0]
    assert out[1] and out[1].endswith(".mp4") and out[2] == "**0 / 12** done"
    first = state["segment_id"]
    refused = b.save(state, None, *groups0, False)  # nothing chosen: refused, nothing saved
    assert "primary activity or the uncertain option" in refused[-1]
    for i in range(3):
        g = copy.deepcopy(groups0)
        g[0] = [cl["groups"][0][1][0]]
        out = b.save(state, cl["primary"][i], *g, False)
        state = out[0]
    assert out[2] == "**3 / 12** done"
    back = b.back(state)  # previous segment, choices pre-filled
    assert back[3] == cl["primary"][2] and back[4] == [cl["groups"][0][1][0]]
    resaved = b.save(back[0], cl["primary"][2], *groups0, True)  # untick one, add uncertain
    skipped = b.skip(resaved[0])
    assert skipped[2] == "**4 / 12** done"  # 3 saved + 1 skipped (the re-save was an already-saved segment)
    rows = [r for r in schema.read_reviews().to_pylist() if r["reviewer"] == "tester-a"]
    assert {r["mode"] for r in rows} == {"blind"}
    actions = [r["action"] for r in rows]
    assert actions.count("reject") == 1 and actions.count("add") == 3 * 2 + 1 + 1  # 3 saves x2, uncertain, skip
    assert any(r["target_id"] == first and r["label_role"] == "primary" for r in rows)
    resumed = BlindController(mock_env["cutter"], labels=labels, pipeline=mock_env["pipeline"]).start(
        "tester-a", "Main sample")
    assert resumed[0]["position"] == 4  # restart resumes at the first unsaved segment


def test_blind_flow_on_fresh_install_without_reviews_db(mock_env, labels):
    """Real installs have run data but no reviews.duckdb (mock data always created one)."""
    from tpa_index.app import build_app
    from tpa_index.paths import output_dir
    from tpa_index.review import checklist

    db = output_dir() / "reviews.duckdb"
    db.unlink()
    assert schema.read_reviews().num_rows == 0 and schema.read_blind_queue().num_rows == 0  # no file: empty
    cl = checklist(labels, mock_env["pipeline"])
    b = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels).blind
    out = b.start("tester-a", "Main sample")  # writes the queue to a brand-new database
    assert out[0]["position"] == 0 and out[2] == "**0 / 12** done"
    assert schema.read_reviews().num_rows == 0  # created empty with the reviews schema
    b.save(out[0], cl["primary"][0], *[[] for _ in cl["groups"]], False)
    resumed = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels).blind.start(
        "tester-a", "Main sample")
    assert resumed[0]["position"] == 1


def test_reviews_db_missing_a_table_reads_empty(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "reviews.duckdb"))  # as written before the fix: blind_queue only
    con.register("e", schema.SCHEMAS["blind_queue"].empty_table())
    con.execute("CREATE TABLE blind_queue AS SELECT * FROM e")
    con.close()
    assert schema.read_reviews(tmp_path).num_rows == 0
    assert schema.read_reviews(tmp_path).schema == schema.SCHEMAS["reviews"]


def test_alternate_vocabulary_shows_in_the_app(mock_env, labels, tmp_path):
    import yaml

    from tpa_index.app import build_app
    from tpa_index.config import load_labels, load_pipeline

    primary_group = mock_env["pipeline"]["moments"]["primary_group"]
    alt = copy.deepcopy(labels)
    for lab in alt["labels"]:
        if lab["group"] == primary_group:
            lab["name"] = "alt_" + lab["name"]
    alt_pipe = copy.deepcopy(mock_env["pipeline"])
    alt_pipe["models"]["motion"]["labels"] = []
    alt_pipe["insights"].update(primary_like=[next(lab["name"] for lab in alt["labels"] if lab["group"] == primary_group)],
                                fusion=[], indicator_rules={})  # 005 rules follow the vocabulary
    alt_pipe["models"]["motion"]["enabled"] = False
    alt_pipe["label_thresholds"] = {}
    (tmp_path / "labels.yaml").write_text(yaml.safe_dump(alt, sort_keys=False))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(alt_pipe, sort_keys=False))
    new_labels = load_labels(tmp_path / "labels.yaml")
    new_pipe = load_pipeline(tmp_path / "pipeline.yaml", labels=new_labels)
    demo = build_app(cutter=mock_env["cutter"], pipeline=new_pipe, labels=new_labels)
    by_id = {c["props"].get("elem_id"): c["props"] for c in _components(demo)}
    shown = [v for _, v in by_id["blind-primary"]["choices"]]
    assert shown and all(v.startswith("alt_") for v in shown)


# ---- US3/US4: review mode (T025, T027) ------------------------------------------------------------------------------
class RecordingCutter:
    """Records the padded clip bounds the app asks for; cuts nothing."""

    def __init__(self, tmp_path):
        self.calls, self.path = [], tmp_path / "clip.mp4"

    def cut(self, video, start_s, end_s, pad_s=0):
        self.calls.append(ClipCutter._bounds(video, start_s, end_s, pad_s))
        return self.path


def _review(mock_env, labels, tmp_path):
    from tpa_index.app import ReviewController

    cutter = RecordingCutter(tmp_path)
    rv = ReviewController(cutter, labels=labels, pipeline=mock_env["pipeline"])
    choices, reason = rv.video_choices()
    assert reason  # blind labeling not done: eval videos hidden, with the reason
    vid = max((v for _, v in choices), key=lambda v: sum(m["video_id"] == v for m in rv.moments))
    return rv, cutter, vid


def test_review_timeline_sorted_and_filters_narrow(mock_env, labels, tmp_path):
    rv, _, vid = _review(mock_env, labels, tmp_path)
    eval_ids = {v["video_id"] for v in schema.read_official("videos").to_pylist() if v["split"] == "eval"}
    assert vid not in eval_ids and not {v for _, v in rv.video_choices()[0]} & eval_ids
    assert rv.timeline(next(iter(eval_ids)))[0] == []  # a locked video id is refused even if sent directly
    table, rows, _ = rv.timeline(vid)
    starts = [r["start_s"] for r in rows]
    assert len(rows) > 1 and starts == sorted(starts) and len(table) == len(rows)
    label = rows[0]["primary_label"]
    narrowed = rv.timeline(vid, label=label)[1]
    from tpa_index.review import ai_labels

    staff = {t for (t, _), r in rv._state().items() if r["new_label"] == label}  # staff edits/adds match too
    assert 0 < len(narrowed) < len(rows)
    assert all(label in {lab for lab, _ in ai_labels(r)} or r["moment_id"] in staff for r in narrowed)
    yes = rv.timeline(vid, needs="yes")[1]
    assert all(r["needs_review"] for r in yes)
    assert len(yes) + len(rv.timeline(vid, needs="no")[1]) == len(rows)
    idle = [r for r in rv.timeline(vid, show_idle=True)[1] if r["kind"] == "idle"]
    assert len(idle) == sum(s["video_id"] == vid and bool(s["is_idle"]) for s in rv.segments)


def test_review_clip_padding_and_unvalidated_badge(mock_env, labels, tmp_path):
    from tpa_index.review import UNVALIDATED

    rv, cutter, vid = _review(mock_env, labels, tmp_path)
    _, rows, _ = rv.timeline(vid)
    duration = next(v["duration_s"] for v in schema.read_official("videos").to_pylist() if v["video_id"] == vid)
    pad = mock_env["pipeline"]["app"]["padding_s"]
    for i, r in enumerate(rows):
        current, clip, panel, ai_choices, history, status, msg = rv.open(rows, i)
        assert cutter.calls[-1] == (max(0.0, r["start_s"] - pad), min(duration, r["end_s"] + pad))
        assert clip == str(cutter.path) and msg == ""
    assert any(c[0] == 0.0 for c in cutter.calls) or rows[0]["start_s"] >= pad
    current, _, panel, ai_choices, _, _, _ = rv.open(rows, 0)
    assert [p[0] for p in panel][0] == current["primary_label"] and ai_choices[0] == current["primary_label"]
    reviewed = set(rv._state())
    for p in panel:
        if (current["moment_id"], p[0]) not in reviewed:
            assert p[3] == UNVALIDATED


def test_review_label_filter_follows_alternate_vocabulary(mock_env, labels, tmp_path):
    import yaml

    from tpa_index.app import ANY, ReviewController
    from tpa_index.config import load_labels, load_pipeline, observation_types

    primary_group = mock_env["pipeline"]["moments"]["primary_group"]
    alt = copy.deepcopy(labels)
    for lab in alt["labels"]:
        if lab["group"] == primary_group:
            lab["name"] = "alt_" + lab["name"]
    alt_pipe = copy.deepcopy(mock_env["pipeline"])
    alt_pipe["models"]["motion"].update(labels=[], enabled=False)
    alt_pipe["insights"].update(primary_like=[next(lab["name"] for lab in alt["labels"] if lab["group"] == primary_group)],
                                fusion=[], indicator_rules={})  # 005 rules follow the vocabulary
    alt_pipe["label_thresholds"] = {}
    (tmp_path / "labels.yaml").write_text(yaml.safe_dump(alt, sort_keys=False))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(alt_pipe, sort_keys=False))
    new_labels = load_labels(tmp_path / "labels.yaml")
    new_pipe = load_pipeline(tmp_path / "pipeline.yaml", labels=new_labels)
    rv = ReviewController(labels=new_labels, pipeline=new_pipe)
    null = new_pipe["calibrate"]["null_group"]
    vocab = [lab["name"] for lab in new_labels["labels"] if lab.get("source", "model") != "rules" and lab["group"] != null]
    assert [v for _, v in rv.label_filter_choices()] == [ANY, *vocab, *observation_types(new_labels)]


def test_review_actions_write_one_row_each(mock_env, labels, tmp_path):
    from tpa_index.config import observation_types
    from tpa_index.review import UNVALIDATED

    rv, _, vid = _review(mock_env, labels, tmp_path)
    _, rows, _ = rv.timeline(vid)
    reviewed = {t for t, _ in rv._state()}
    i = next(i for i, r in enumerate(rows) if r["moment_id"] not in reviewed and r["secondary_labels"])
    current, _, panel, ai_choices, _, _, _ = rv.open(rows, i)
    assert all(p[3] == UNVALIDATED for p in panel)

    def count():
        return schema.read_reviews().num_rows

    n0 = count()
    lab = ai_choices[0]
    assert rv.act(current, "", lab, "accept")[-1].startswith("Enter a reviewer") and count() == n0  # reviewer needed
    out = rv.act(current, "staff-a", lab, "accept")
    assert count() == n0 + 1 and out[0][0][3].startswith("accept by staff-a")
    assert "partly" in out[2]
    edit_to = rv.edit_choices(current, ai_choices[1])[0]
    assert rv.act(current, "staff-a", ai_choices[1], "edit")[-1] == "edit needs the new label" and count() == n0 + 1
    rv.act(current, "staff-a", ai_choices[1], "edit", edit_to=edit_to, note="closer match")
    assert count() == n0 + 2
    for action in ("reject", "defer"):
        rv.act(current, "staff-a", ai_choices[-1], action)
    assert count() == n0 + 4
    assert "already an AI suggestion" in rv.add_label(current, "staff-a", ai_choices[1])[-1]
    assert count() == n0 + 4  # adding an AI label would hide its edit: refused
    out = rv.add_label(current, "staff-a", next(n for n in rv.vocab if n not in ai_choices))
    obs = observation_types(labels)[0]
    out = rv.add_observation(current, "staff-a", obs, note="lights flicker")
    assert count() == n0 + 6
    mine = [r for r in schema.read_reviews().to_pylist() if r["target_id"] == current["moment_id"]
            and r["reviewer"] == "staff-a"]
    assert [r["action"] for r in mine] == ["accept", "edit", "reject", "defer", "add", "add"]
    assert {r["mode"] for r in mine} == {"review"} and mine[1]["new_label"] == edit_to
    assert mine[-1]["label_role"] == "observation" and mine[-1]["note"] == "lights flicker"
    assert [h[3] for h in out[1]][-6:] == ["accept", "edit", "reject", "defer", "add", "add"]  # history refreshed


def test_review_tab_reminder_and_observation_choices(mock_env, labels):
    from tpa_index.app import REMINDER, build_app
    from tpa_index.config import observation_types

    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
    by_id = {c["props"].get("elem_id"): c["props"] for c in _components(demo)}
    assert by_id["review-reminder"]["value"] == REMINDER and "No names. No judgments about a person." in REMINDER
    assert [v for _, v in by_id["review-add-obs"]["choices"]] == observation_types(labels)


def test_reviewer_names_are_case_insensitive(mock_env, labels):
    from tpa_index.app import BlindController
    from tpa_index.review import eval_unlocked, load_queue

    b = BlindController(mock_env["cutter"], labels=labels, pipeline=mock_env["pipeline"])
    groups = [[] for _ in b.checklist["groups"]]
    out = b.start("  Ana ", "Main sample")
    for _ in range(2):
        out = b.save(out[0], b.checklist["primary"][0], *groups, False)
    assert b.start("ana", "Main sample")[0]["position"] == 2  # same reviewer: resumes
    out = b.start("Ana", "Overlap set (second labeler)")
    while out[0]["position"] is not None:
        out = b.save(out[0], b.checklist["primary"][0], *groups, False)
    assert b.start("ANA", "Overlap set (second labeler)")[0]["position"] is None  # already finished
    reviewers = {r["reviewer"] for r in schema.read_reviews().to_pylist() if r["mode"] == "blind"}
    assert reviewers == {"ana"}
    _, reason = eval_unlocked()
    assert f"{len(load_queue('overlap'))} overlap" in reason  # one reviewer, not two


def test_note_cleared_after_save_kept_on_error(mock_env, labels, tmp_path):
    from tpa_index.app import build_app
    from tpa_index.config import observation_types

    demo = build_app(cutter=RecordingCutter(tmp_path), pipeline=mock_env["pipeline"], labels=labels)
    note_id = next(b._id for b in demo.blocks.values() if getattr(b, "elem_id", None) == "review-note")
    obs_btn = next(b._id for b in demo.blocks.values() if getattr(b, "elem_id", None) == "review-obs")
    fn = next(f for f in demo.fns.values() if any(t[0] == obs_btn for t in f.targets)
              and note_id in [o._id for o in f.outputs])
    rv = demo.review
    vid = rv.video_choices()[0][0][1]
    current = rv.open(rv.timeline(vid)[1], 0)[0]
    obs = observation_types(labels)[0]
    refused = fn.fn(current, "", obs, "keep me")  # no reviewer: nothing saved
    assert refused[-1] != "" and "Enter a reviewer" in refused[3]
    ok = fn.fn(current, "staff-a", obs, "a note")
    assert ok[-1] == "" and ok[3].startswith("Saved")


# ---- US5: local only, no outbound connections, no lasting clip copies (T017-T018) -----------------------------
def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_launch_is_loopback_only_and_leaves_nothing(mock_env, labels, monkeypatch):
    import ipaddress
    import os
    import urllib.request

    import psutil

    from tpa_index.app import build_app, launch_kwargs

    session = mock_env["cutter"].session
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(session))  # as launch() does
    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
    port = _free_port()
    kwargs = launch_kwargs(session, port)
    assert kwargs["server_name"] == "127.0.0.1" and kwargs["share"] is False and kwargs["enable_monitoring"] is False
    demo.launch(**kwargs, prevent_thread_lock=True)
    try:
        assert demo.server_name == "127.0.0.1" and not demo.share
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
            assert resp.status == 200
        out = demo.blind.start("tester-a", "Main sample")
        video = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "blind-clip")
        assert video.move_resource_to_block_cache(out[1]) == out[1]  # served in place: no second copy
        out = demo.blind.save(out[0], demo.blind.checklist["primary"][0], *[[] for _ in demo.blind.checklist["groups"]],
                              False)
        assert video.move_resource_to_block_cache(out[1]) == out[1]
        assert len(list(session.rglob("*.mp4"))) <= 2  # shown clip + one prefetch; the previous clip was deleted
        remote = [c.raddr.ip for c in psutil.Process(os.getpid()).net_connections(kind="inet") if c.raddr]
        assert all(ipaddress.ip_address(ip).is_loopback for ip in remote), remote
        listening = [c.laddr.ip for c in psutil.Process(os.getpid()).net_connections(kind="inet")
                     if c.status == psutil.CONN_LISTEN]
        assert listening and all(ipaddress.ip_address(ip).is_loopback for ip in listening), listening
    finally:
        demo.close()
    mock_env["cutter"].cleanup()
    assert not session.exists()


def test_share_never_enabled_in_source():
    from tpa_index.paths import repo_root

    for path in (repo_root() / "src").rglob("*.py"):
        assert "share=True" not in path.read_text(), path.name


def test_clip_errors_shown_to_staff_are_redacted(mock_env, labels):
    from tpa_index.app import build_app
    from tpa_index.paths import data_dir

    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
    b = demo.blind
    first = b.start("tester-a", "Main sample")[0]
    from tpa_index.review import load_queue

    video_id = load_queue("main")[0]["video_id"]
    b.videos[video_id] = {**b.videos[video_id], "filename": "missing/secret_take_9.mp4"}
    out = b.replay(first)
    assert out[1] is None and "Use Skip" in out[-1]
    assert str(data_dir()) not in out[-1] and "secret_take_9" not in out[-1]


# ---- US2: overlap and top-up in the app (T021) -----------------------------------------------------------------
def test_second_reviewer_gets_the_overlap_set_in_order(mock_env, labels):
    from tpa_index.app import build_app
    from tpa_index.review import load_queue

    b = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels).blind
    b.start("tester-a", "Main sample")
    seen = []
    out = b.start("tester-b", "Overlap set (second labeler)")
    while out[0]["position"] is not None:
        seen.append(out[0]["segment_id"])
        out = b.save(out[0], b.checklist["primary"][0], *[[] for _ in b.checklist["groups"]], False)
    assert seen == [q["segment_id"] for q in load_queue("overlap")] and len(seen) == 4
    assert seen == [q["segment_id"] for q in load_queue("main")][:4]


def test_topup_hidden_until_predict_runs_and_never_shows_target(mock_env, labels):
    import json

    from tpa_index.app import build_app
    from tpa_index.review import load_queue

    run_id = mock_env["official"].read_text().split("[")[1].split("]")[0]
    mock_env["official"].write_text(f"segment: [{run_id}]\n")  # no official predict runs yet
    b = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels).blind
    b.start("tester-a", "Main sample")
    names, note = b.queue_choices()
    assert "Top-up" not in names and "official predict runs" in note
    mock_env["official"].write_text(f"predict: [{run_id}]\n")
    names, note = b.queue_choices()
    assert "Top-up" in names and note == ""
    out = b.start("tester-a", "Top-up")
    topup = load_queue("topup")
    assert topup and out[0]["segment_id"] == topup[0]["segment_id"]
    targets = {q["target_label"] for q in topup}
    shown = json.dumps([out[0], *out[2:]], default=str)  # everything the screen receives except the clip file
    assert "target_label" not in shown and all(f'"{t}"' not in json.dumps(out[0]) for t in targets)



def test_empty_topup_says_why_not_finished(mock_env, labels, pipeline):
    from tpa_index.app import build_app

    big = copy.deepcopy(mock_env["pipeline"])
    big["app"].update(sample_size=100000, overlap_size=4)  # every eval segment lands in the main sample
    b = build_app(cutter=mock_env["cutter"], pipeline=big, labels=labels).blind
    b.start("tester-a", "Main sample")
    out = b.start("tester-a", "Top-up")
    assert out[0]["position"] is None and "no segments" in out[-1] and "finished" not in out[-1]


def test_shortcuts_click_the_input_not_the_label():
    """A label click toggles a checkbox on screen but not Gradio's checkbox-group value (found in Checkpoint 1)."""
    from tpa_index.app import PAGE_JS

    assert "lab.querySelector('input')" in PAGE_JS and "{ lab.click();" not in PAGE_JS
