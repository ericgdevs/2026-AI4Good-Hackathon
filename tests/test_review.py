import copy
from datetime import datetime, timedelta, timezone

import yaml

from tpa_index import config
from tpa_index.config import load_labels, load_pipeline
from tpa_index.review import checklist, key_map, latest_state, role_of


def _names(labels, **where):
    return [lab["name"] for lab in labels["labels"] if all(lab.get(k, "model") == v for k, v in where.items())]


def test_checklist_follows_config(labels, pipeline):
    cl = checklist(labels, pipeline)
    primary_group = pipeline["moments"]["primary_group"]
    assert cl["primary"] == [lab["name"] for lab in labels["labels"] if lab["group"] == primary_group]
    assert [g for g, _ in cl["groups"]] == [g for g in pipeline["app"]["blind_groups"] if g != primary_group]
    shown = set(cl["primary"]) | {n for _, names in cl["groups"] for n in names}
    assert not shown & set(_names(labels, source="rules"))  # never rules flags
    null = pipeline["calibrate"]["null_group"]
    assert not shown & {lab["name"] for lab in labels["labels"] if lab["group"] == null}  # never the noise group
    staff = set(_names(labels, source="staff"))
    assert staff and staff <= shown  # staff-only labels are exactly what blind labeling needs
    assert cl["uncertain"] == pipeline["app"]["uncertain_label"]


def test_every_checklist_label_gets_a_key(labels, pipeline):
    cl = checklist(labels, pipeline)
    keys, unkeyed = key_map(cl)
    assert unkeyed == []
    assert list(keys.values())[:len(cl["primary"])] == cl["primary"] and keys["1"] == cl["primary"][0]
    assert "u" not in keys and len(set(keys.values())) == len(keys)


def test_roles(labels, pipeline):
    cl = checklist(labels, pipeline)
    first = cl["primary"][0]
    indicator_group = pipeline["app"]["indicator_groups"][0]
    ind = next(lab["name"] for lab in labels["labels"] if lab["group"] == indicator_group)
    assert role_of(first, labels, pipeline, chosen_primary=first) == "primary"
    assert role_of(cl["primary"][1], labels, pipeline, chosen_primary=first) == "secondary"
    assert role_of(ind, labels, pipeline) == "indicator"
    assert role_of(cl["uncertain"], labels, pipeline) == "secondary"


def test_latest_state_picks_newest_per_target_label():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"target_id": "s1", "label": "x", "created_at": t0 + timedelta(minutes=m), "review_id": rid, "mode": mode,
             "action": act} for m, rid, mode, act in ((0, "a", "blind", "add"), (2, "b", "blind", "reject"),
                                                       (1, "c", "review", "accept"))]
    assert latest_state(rows)[("s1", "x")]["action"] == "reject"
    assert latest_state(rows, mode="review")[("s1", "x")]["action"] == "accept"


def test_vocabulary_swap(tmp_path, labels, pipeline):
    """A different labels.yaml: the checklist, keys and roles follow it; no shipped label survives."""
    primary_group = pipeline["moments"]["primary_group"]
    indicator_group = pipeline["app"]["indicator_groups"][0]
    rules_group = next(lab["group"] for lab in labels["labels"] if lab.get("source") == "rules")
    null = pipeline["calibrate"]["null_group"]
    alt = copy.deepcopy(labels)
    keep = [lab for lab in alt["labels"] if lab["group"] not in (primary_group, indicator_group, rules_group)]
    alt["labels"] = keep + [
        {"name": "juggling", "group": primary_group, "prompts": ["a", "b", "c"], "color": "#000001"},
        {"name": "drumming", "group": primary_group, "prompts": ["d", "e", "f"], "color": "#000002"},
        {"name": "lifts_peer", "group": indicator_group, "source": "staff", "color": "#000003"},
        {"name": "unsure", "group": rules_group, "source": "rules", "color": "#000004"},
        {"name": "flagged", "group": rules_group, "source": "rules", "color": "#000005"},
    ]
    other_groups = [g for g in config.LABEL_GROUPS if g not in (primary_group, indicator_group, rules_group, null)]
    venue = other_groups[0]  # reuse an existing group name as the "new" checklist group
    for lab in alt["labels"]:
        if lab["group"] == venue:
            lab["name"] = "alt_" + lab["name"]
    (tmp_path / "labels.yaml").write_text(yaml.safe_dump(alt, sort_keys=False))
    alt_pipe = copy.deepcopy(pipeline)
    alt_pipe["app"]["blind_groups"] = [primary_group, venue, indicator_group]
    alt_pipe["app"]["uncertain_label"] = "unsure"
    alt_pipe["models"]["motion"]["labels"] = ["juggling"]
    alt_pipe["label_thresholds"] = {}
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(alt_pipe, sort_keys=False))

    new_labels = load_labels(tmp_path / "labels.yaml")
    new_pipe = load_pipeline(tmp_path / "pipeline.yaml", labels=new_labels)
    cl = checklist(new_labels, new_pipe)
    assert cl["primary"] == ["juggling", "drumming"] and cl["uncertain"] == "unsure"
    assert [g for g, _ in cl["groups"]] == [venue, indicator_group]
    assert dict(cl["groups"])[indicator_group] == ["lifts_peer"]
    keys, _ = key_map(cl)
    assert keys["1"] == "juggling" and keys["2"] == "drumming"
    assert role_of("lifts_peer", new_labels, new_pipe) == "indicator"
    shipped = {lab["name"] for lab in labels["labels"]}
    shown = set(cl["primary"]) | {n for _, names in cl["groups"] for n in names} | {cl["uncertain"]}
    assert not shown & shipped


def test_no_label_or_group_names_hardcoded():
    """The vocabulary changes; 003 modules must read every label and group from config."""
    import re

    from tpa_index.paths import repo_root

    shipped = load_labels()
    names = {lab["name"] for lab in shipped["labels"]} | {lab["group"] for lab in shipped["labels"]}
    names -= set(schema.SCHEMAS["moments"].names)  # fixed table columns (e.g. moments.needs_review), not vocabulary
    for module in ("review.py", "clips.py", "app.py"):
        path = repo_root() / "src/tpa_index" / module
        if not path.exists():
            continue
        source = path.read_text()
        found = sorted(n for n in names if re.search(rf"""['"]{re.escape(n)}['"]""", source))
        assert not found, f"{module} hardcodes {found}"


# ---- blind rows (T011) --------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from tpa_index.review import blind_rows, previous_blind, skip_row  # noqa: E402
from tpa_index.schema import append_reviews, validate  # noqa: E402

SEG = {"segment_id": "mock04_0"}


def _pick(labels, pipeline):
    cl = checklist(labels, pipeline)
    indicator_group = pipeline["app"]["indicator_groups"][0]
    other = next(names for g, names in cl["groups"] if g != indicator_group)[0]
    ind = dict(cl["groups"])[indicator_group][0]
    return cl["primary"][0], other, ind


def test_blind_rows_roles_and_actions(labels, pipeline):
    primary, other, ind = _pick(labels, pipeline)
    rows = blind_rows(SEG, primary, [other, ind], True, "mock-staff-a", labels, pipeline)
    by = {r["label"]: r for r in rows}
    assert by[primary]["label_role"] == "primary" and by[other]["label_role"] == "secondary"
    assert by[ind]["label_role"] == "indicator" and pipeline["app"]["uncertain_label"] in by
    assert all(r["action"] == "add" and r["new_label"] == r["label"] and r["mode"] == "blind"
               and r["source"] == "staff" and r["guideline_version"] == labels["guideline_version"] for r in rows)
    assert validate("reviews", [dict(r, review_id=str(i), created_at=datetime.now(timezone.utc))
                                for i, r in enumerate(rows)]).num_rows == 4


def test_save_needs_primary_or_uncertain(labels, pipeline):
    with pytest.raises(ValueError, match="primary activity or the uncertain option"):
        blind_rows(SEG, None, [], False, "mock-staff-a", labels, pipeline)
    assert blind_rows(SEG, None, [], True, "mock-staff-a", labels, pipeline)


def test_relabel_writes_only_changes(labels, pipeline):
    primary, other, ind = _pick(labels, pipeline)
    append_reviews(blind_rows(SEG, primary, [other, ind], False, "mock-staff-a", labels, pipeline))
    prev = previous_blind(SEG["segment_id"], "mock-staff-a")
    again = blind_rows(SEG, primary, [other], False, "mock-staff-a", labels, pipeline, previous=prev)
    assert [(r["label"], r["action"], r["new_label"]) for r in again] == [(ind, "reject", None)]
    assert blind_rows(SEG, primary, [other, ind], False, "mock-staff-a", labels, pipeline, previous=prev) == []
    assert previous_blind(SEG["segment_id"], "mock-staff-b") == {}  # other reviewers never mix in


def test_skip_row_is_an_observation(labels, pipeline):
    r = skip_row(SEG, "mock-staff-a", labels, pipeline)
    assert (r["label_role"], r["action"], r["label"]) == ("observation", "add", pipeline["app"]["skip_observation"])


# ---- queues (T012) ------------------------------------------------------------------------------------------------
from tpa_index import schema  # noqa: E402
from tpa_index.mock import generate  # noqa: E402
from tpa_index.review import build_main_queues, eval_segments, load_queue, next_position, progress, queue_missing  # noqa: E402


@pytest.fixture
def mock_official(tmp_path, monkeypatch):
    generate(seed=0)
    path = tmp_path / "official.yaml"
    path.write_text("predict: [mock-seed0]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    return path


def _app(pipeline, **kw):
    return {**pipeline["app"], **kw}


def test_main_queue_is_stable_eval_only_and_written_once(mock_official, pipeline):
    app = _app(pipeline, sample_size=40, overlap_size=10)
    assert build_main_queues(app) == 40
    first = [q["segment_id"] for q in load_queue("main")]
    assert build_main_queues(app) == 40 and [q["segment_id"] for q in load_queue("main")] == first
    assert len(schema.read_blind_queue()) == 50  # second call wrote nothing
    eval_ids = {s["segment_id"] for s in eval_segments()}
    assert set(first) <= eval_ids
    assert [q["segment_id"] for q in load_queue("overlap")] == first[:10]


def test_same_runs_and_seed_give_same_order_in_a_fresh_database(mock_official, pipeline, tmp_path):
    app = _app(pipeline, sample_size=40, overlap_size=10)
    build_main_queues(app)
    other = tmp_path / "other-out"
    generate(other, seed=0)
    build_main_queues(app, root=other)
    assert [q["segment_id"] for q in load_queue("main")] == \
        [q["segment_id"] for q in load_queue("main", root=other)]


def test_sample_larger_than_eval_set_takes_all(mock_official, pipeline):
    n = build_main_queues(_app(pipeline, sample_size=100000, overlap_size=10))
    assert n == len(eval_segments())


def test_main_sample_is_shared_overlap_is_per_reviewer(mock_official, pipeline, labels):
    from tpa_index.review import blind_rows

    build_main_queues(_app(pipeline, sample_size=20, overlap_size=5))
    queue = load_queue("main")
    primary = checklist(labels, pipeline)["primary"][0]

    def label(items, reviewer):
        for q in items:
            append_reviews(blind_rows(q, primary, [], False, reviewer, labels, pipeline))

    label(queue[:5], "mock-staff-a")  # A labels positions 0-4
    assert next_position("main", "mock-staff-a") == 5
    assert next_position("main", "mock-staff-b") == 5  # B continues where the team stopped
    assert next_position("overlap", "mock-staff-b") == 0  # overlap: B labels the first items themselves
    assert progress("overlap", "mock-staff-b") == (0, 5) and progress("overlap", "mock-staff-a") == (5, 5)
    label(queue[5:8], "mock-staff-b")
    assert progress("main", "mock-staff-a") == (8, 20) == progress("main", "mock-staff-b")  # team total
    assert next_position("main", "mock-staff-a") == 8
    label(queue[8:], "mock-staff-a")
    assert next_position("main", "mock-staff-b") is None


def test_segment_stage_fallback(tmp_path, monkeypatch, pipeline):
    generate(seed=0)
    path = tmp_path / "official.yaml"
    path.write_text("segment: [mock-seed0]\n")  # no predict runs official yet
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    assert build_main_queues(_app(pipeline, sample_size=10, overlap_size=5)) == 10


def test_changed_official_runs_never_reshuffle(mock_official, pipeline):
    build_main_queues(_app(pipeline, sample_size=10, overlap_size=5))
    before = [q["segment_id"] for q in load_queue("main")]
    mock_official.write_text("predict: []\n")  # the run is no longer official
    assert [q["segment_id"] for q in load_queue("main")] == before
    assert sorted(queue_missing("main")) == sorted(before)


# ---- top-up (T020) ----------------------------------------------------------------------------------------------
from tpa_index.review import build_topup_queue, topup_available  # noqa: E402


def test_topup_hidden_without_predict_runs(tmp_path, monkeypatch, pipeline, labels):
    from tpa_index.config import ConfigError

    generate(seed=0)
    path = tmp_path / "official.yaml"
    path.write_text("segment: [mock-seed0]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    build_main_queues(_app(pipeline, sample_size=10, overlap_size=5))
    ok, reason = topup_available()
    assert not ok and "official predict runs" in reason
    with pytest.raises(ConfigError):
        build_topup_queue(_app(pipeline, sample_size=10, overlap_size=5), labels, pipeline)


def test_topup_cap_exclusions_shuffle_and_tags(mock_official, pipeline, labels):
    from tpa_index.review import blind_rows

    app = _app(pipeline, sample_size=10, overlap_size=5, topup_per_label=3, rare_min_positives=2)
    build_main_queues(app)
    main = load_queue("main")
    common = checklist(labels, pipeline)["primary"][0]
    for q in main[:3]:  # common label gets 3 blind positives: not rare
        append_reviews(blind_rows(q, common, [], False, "tester-a", labels, pipeline))
    assert topup_available()[0]
    n = build_topup_queue(app, labels, pipeline)
    topup = load_queue("topup")
    assert n == len(topup) > 0
    per_label = {}
    for q in topup:
        per_label[q["target_label"]] = per_label.get(q["target_label"], 0) + 1
    assert max(per_label.values()) <= 3 and common not in per_label
    assert not {q["segment_id"] for q in topup} & {q["segment_id"] for q in main}
    assert all(q["target_label"] for q in topup)
    assert build_topup_queue(app, labels, pipeline) == n and len(load_queue("topup")) == n  # written once
    # order is shuffled, not by score
    preds = {(p["segment_id"], p["label"]): p["score"] for p in schema.read_official("predictions").to_pylist()}
    scores = [preds[(q["segment_id"], q["target_label"])] for q in topup]
    assert scores != sorted(scores, reverse=True) or len(scores) < 3


# ---- review mode: status, filters, unlock (T023) -------------------------------------------------------------------
from tpa_index.review import (UNVALIDATED, ai_labels, eval_unlocked, filter_moments, label_panel,  # noqa: E402
                              moment_status, observation_row, review_row, review_videos)


def _moment(labels, pipeline, mid="m1", start=0.0, needs=False, secondary=None):
    cl = checklist(labels, pipeline)
    sec = cl["groups"][0][1][:2] if secondary is None else secondary
    return {"moment_id": mid, "video_id": "v1", "start_s": start, "end_s": start + 8, "primary_label": cl["primary"][0],
            "secondary_labels": sec, "indicators": [], "confidence": 0.7, "needs_review": needs, "run_id": "r"}


def _state(moment, actions, labels):
    """actions: {label: action}; review rows in insertion order, via review_row."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    role = dict(ai_labels(moment))
    for i, (label, action) in enumerate(actions):
        r = review_row(moment, label, role.get(label, "secondary"), action, "staff-a", labels,
                       new_label=label if action == "edit" else None)
        rows.append({**r, "created_at": t0 + timedelta(minutes=i), "review_id": f"{i:04d}"})
    return latest_state(rows, "review")


def test_moment_status_cases(labels, pipeline):
    m = _moment(labels, pipeline)
    names = [lab for lab, _ in ai_labels(m)]
    assert len(names) == 3
    assert moment_status(m, {}) == "unreviewed"
    assert moment_status(m, _state(m, [(names[0], "accept")], labels)) == "partly"
    assert moment_status(m, _state(m, [(names[0], "accept"), (names[1], "defer")], labels)) == "deferred"
    done = [(names[0], "accept"), (names[1], "reject"), (names[2], "edit")]
    assert moment_status(m, _state(m, done, labels)) == "done"
    assert moment_status(m, _state(m, done + [(names[2], "defer")], labels)) == "deferred"  # latest row counts
    # a staff add on another label does not review the AI labels
    other = next(n for n in checklist(labels, pipeline)["primary"] if n not in names)
    assert moment_status(m, _state(m, [(other, "add")], labels)) == "unreviewed"


def test_combined_filters_sorted_by_time(labels, pipeline):
    cl = checklist(labels, pipeline)
    a = _moment(labels, pipeline, "a", start=16, needs=True)
    b = _moment(labels, pipeline, "b", start=0, needs=True)
    c = _moment(labels, pipeline, "c", start=8, needs=False, secondary=[])
    state = _state(a, [(a["primary_label"], "accept")], labels)
    assert [m["moment_id"] for m in filter_moments([a, b, c], state)] == ["b", "c", "a"]
    assert [m["moment_id"] for m in filter_moments([a, b, c], state, needs_review=True)] == ["b", "a"]
    assert [m["moment_id"] for m in filter_moments([a, b, c], state, needs_review=True, status="partly")] == ["a"]
    sec = cl["groups"][0][1][0]
    assert [m["moment_id"] for m in filter_moments([a, b, c], state, label=sec, status="unreviewed")] == ["b"]
    obs = labels["observation_types"][0]["name"]
    state = {**state, **latest_state([{**observation_row(c, obs, "staff-a", labels), "review_id": "z",
                                       "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc)}], "review")}
    assert [m["moment_id"] for m in filter_moments([a, b, c], state, label=obs)] == ["c"]  # staff-added match


def test_retired_labels_show_and_panel_orders_by_strength(labels, pipeline):
    m = _moment(labels, pipeline, secondary=[])
    s1, s2 = checklist(labels, pipeline)["groups"][0][1][:2]
    m["secondary_labels"] = [s1, s2, "gone_label"]
    conf = {s1: 0.02, s2: 0.3, "gone_label": 0.5}
    thr = {m["primary_label"]: 0.5, s1: 0.001, s2: 0.25}  # s1: 20x its threshold; s2: 1.2x
    rows = label_panel(m, {}, labels, conf, thr)
    assert [r["label"] for r in rows] == [m["primary_label"], s1, s2, "gone_label"]
    assert rows[0]["confidence"] == m["confidence"]  # primary shows the moment's own confidence
    assert [r["retired"] for r in rows] == [False, False, False, True]
    assert all(r["badge"] == UNVALIDATED for r in rows)
    st = _state(m, [(s1, "reject")], labels)
    assert label_panel(m, st, labels, conf, thr)[1]["badge"].startswith("reject by staff-a")


def _blind(segment_ids, reviewer, labels, pipeline):
    primary = checklist(labels, pipeline)["primary"][0]
    for sid in segment_ids:
        append_reviews(blind_rows({"segment_id": sid}, primary, [], False, reviewer, labels, pipeline))


def test_eval_unlock_needs_main_and_two_overlap_reviewers(mock_official, pipeline, labels):
    eval_ids = {v["video_id"] for v in schema.read_official("videos").to_pylist() if v["split"] == "eval"}
    assert eval_ids
    assert not eval_unlocked()[0]  # no queue yet
    assert not {v["video_id"] for v in review_videos()} & eval_ids
    build_main_queues(_app(pipeline, sample_size=10, overlap_size=3))
    main = [q["segment_id"] for q in load_queue("main")]
    overlap = [q["segment_id"] for q in load_queue("overlap")]
    _blind(main, "staff-a", labels, pipeline)
    ok, reason = eval_unlocked()
    assert not ok and "3 overlap" in reason  # main done, overlap has one reviewer
    _blind(overlap, "staff-a", labels, pipeline)  # same reviewer again: still one distinct reviewer
    assert not eval_unlocked()[0]
    assert not {v["video_id"] for v in review_videos()} & eval_ids
    _blind(overlap, "staff-b", labels, pipeline)
    assert eval_unlocked() == (True, "")
    assert eval_ids <= {v["video_id"] for v in review_videos()}


def test_overlap_alone_does_not_unlock(mock_official, pipeline, labels):
    build_main_queues(_app(pipeline, sample_size=10, overlap_size=3))
    overlap = [q["segment_id"] for q in load_queue("overlap")]
    _blind(overlap, "staff-a", labels, pipeline)
    _blind(overlap, "staff-b", labels, pipeline)
    ok, reason = eval_unlocked()
    assert not ok and "7 main-sample" in reason


# ---- review rows (T026) --------------------------------------------------------------------------------------------
def test_review_rows_validate_and_latest_state(labels, pipeline):
    m = _moment(labels, pipeline)
    lab, role = ai_labels(m)[1]
    other = checklist(labels, pipeline)["groups"][0][1][-1]
    obs = labels["observation_types"][0]["name"]
    rows = [review_row(m, lab, role, a, "staff-a", labels) for a in ("accept", "reject", "defer")]
    rows += [review_row(m, lab, role, "edit", "staff-a", labels, new_label=other, note="n"),
             review_row(m, other, role_of(other, labels, pipeline), "add", "staff-a", labels),
             observation_row(m, obs, "staff-a", labels, note="lights")]
    tbl = validate("reviews", [{**r, "review_id": f"{i}", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
                               for i, r in enumerate(rows)])
    assert set(tbl["mode"].to_pylist()) == {"review"} and set(tbl["target_type"].to_pylist()) == {"moment"}
    assert rows[4]["source"] == "staff" and rows[4]["new_label"] == other
    assert rows[5]["label_role"] == "observation" and rows[5]["new_label"] == obs
    with pytest.raises(ValueError, match="edit needs"):
        review_row(m, lab, role, "edit", "staff-a", labels)
    with pytest.raises(ValueError, match="no new label"):
        review_row(m, lab, role, "accept", "staff-a", labels, new_label=other)
    with pytest.raises(ValueError, match="unknown observation"):
        observation_row(m, "not_an_observation", "staff-a", labels)
    st = _state(m, [(lab, "accept"), (lab, "defer"), (lab, "edit")], labels)
    assert st[(m["moment_id"], lab)]["action"] == "edit"


def test_blind_checklist_identical_to_0_4_0(labels, pipeline):
    """SC-012: the 0.5.0 labels (audio label singing_voice added) give the 0.4.0 blind checklist,
    so 0.4.0 and 0.5.0 blind reviews form one eval set."""
    import json

    from tpa_index.paths import repo_root

    snapshot = json.loads((repo_root() / "tests/data/blind_checklist_0.4.0.json").read_text())
    assert snapshot["guideline_version"] == "0.4.0"
    assert labels["guideline_version"] == "0.5.0"
    assert json.loads(json.dumps(checklist(labels, pipeline))) == snapshot["checklist"]
