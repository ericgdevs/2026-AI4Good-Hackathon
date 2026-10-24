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
    page = json.dumps(_components(demo), default=str)
    model_names = [m.rsplit("/", 1)[-1] for m in mock_env["pipeline"]["models"]["main"].values()]
    for word in ("AI suggestion", "score", "confidence", *model_names):
        assert word not in page, word
    assert by_id["reviewer"]["placeholder"] == "staff first name or role"


def test_blind_flow_never_reads_model_output(mock_env, labels, monkeypatch):
    from tpa_index.app import build_app
    from tpa_index.review import checklist

    real = schema.read_official

    def guarded(table, *a, **k):
        if table in ("predictions", "moments"):
            raise AssertionError(f"blind mode read {table}")
        return real(table, *a, **k)

    monkeypatch.setattr(schema, "read_official", guarded)
    demo = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels)
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
    resumed = build_app(cutter=mock_env["cutter"], pipeline=mock_env["pipeline"], labels=labels).blind.start(
        "tester-a", "Main sample")
    assert resumed[0]["position"] == 4  # restart resumes at the first unsaved segment


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
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(session / "gradio"))
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
        demo.blind.save(out[0], demo.blind.checklist["primary"][0], *[[] for _ in demo.blind.checklist["groups"]], False)
        remote = [c.raddr.ip for c in psutil.Process(os.getpid()).net_connections(kind="inet") if c.raddr]
        assert all(ipaddress.ip_address(ip).is_loopback for ip in remote), remote
        listening = [c.laddr.ip for c in psutil.Process(os.getpid()).net_connections(kind="inet")
                     if c.status == psutil.CONN_LISTEN]
        assert listening and all(ipaddress.ip_address(ip).is_loopback for ip in listening), listening
    finally:
        demo.close()
    mock_env["cutter"].cleanup()
    assert not session.exists() and not (session / "gradio").exists()


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
