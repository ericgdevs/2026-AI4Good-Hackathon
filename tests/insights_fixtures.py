"""A mock world for 005 (evaluate / insights): official runs for every stage plus blind labels. No media, no data.

Videos (120 s, 8 s windows, 2 s overlap -> 20 segments `{vid}_{start_ms}`):
  d01 dev/early, e01 eval/early, e02 eval/showcase, x01 dev/unknown.

Per-segment pattern (index i in the video; scaled worlds repeat it every 20 segments):
  0-4   singing on stage (performance, participant); person_count 1 at 0-1 (solo), 3 at 2-4; singing_voice audio.
        e01 seg 4 is the planted fusion case: visual singing 0.9 but AST singing_voice and music 0.1.
  5-9   dance (performance, participant), person_count 4, music audio. In e01 the dance score is 0.55, so that
        moment is needs_review (confidence below threshold * 1.2).
  10-12 idle (is_idle, everything low): never in a moment -> "model made no call".
  13-15 instruction (instruction + actor_staff_facilitation, participant low): no activity clears.
  16-18 spoken word on stage (lyricism_spoken_word, performance, participant), speech audio, has_speech.
  19    audience_watching with applause 0.8, right after the spoken-word performance.
  e02 10-14 are overridden by the planted staff stretch (actor_staff_facilitation 0.9, participant 0.1).
  Faces: performance segments have 2 faces; e02 faces have `positive` as argmax, others `neutral`.

Planted leaks (must never reach an output): filename mock/secret_name.mp4 on e01; a transcript.json with
ZQXJ-005-SENTINEL in the transcript run folder.

Blind labels (reviewer mock-staff-a), main queue = e01 0-19, e02 0-9, d01 0-2; topup queue = e02 15-17:
  mostly agree with the model; e02 3 human primary dance (model says singing -> failure gallery);
  e01 5 ticks two categories; e01 6 ticks none; e02 9 saved with Skip; e01 0 had `dance` added then rejected;
  idle 10-12 and instruction 13-15 saved as uncertain (13-15 also tick instruction + actor_staff_facilitation).
"""
import json
from datetime import datetime, timezone

import yaml

from tpa_index import config, schema
from tpa_index.config import load_labels, load_pipeline, model_labels
from tpa_index.moments import merge_moments
from tpa_index.review import blind_rows, skip_row
from tpa_index.schema import append_blind_queue, append_reviews, write_table
from tpa_index.score import rules_rows

SIGLIP, AST, ENET = "mock-siglip", "mock-ast", "mock-enet"
SECRET, SENTINEL = "secret_name", "ZQXJ-005-SENTINEL"
AUDIO = ["singing_voice", "music", "speech", "applause", "silence"]
EXPR = ["positive", "neutral", "negative", "upset"]
T0 = datetime(2026, 10, 24, tzinfo=timezone.utc)
VIDEOS = [("d01", "dev", "early"), ("e01", "eval", "early"), ("e02", "eval", "showcase"), ("x01", "dev", "unknown")]


def _pattern(i):
    """(visual scores, audio scores, person_count, faces, has_speech, is_idle) for pattern index i (0-19)."""
    v, a = {}, {}
    if i <= 4:
        v = {"singing": 0.9, "performance": 0.9, "actor_participant": 0.9}
        a = {"singing_voice": 0.8, "music": 0.3}
        return v, a, (1 if i <= 1 else 3), 2, False, False
    if i <= 9:
        v = {"dance": 0.9, "performance": 0.8, "actor_participant": 0.9}
        return v, {"music": 0.8}, 4, 2, False, False
    if i <= 12:
        return {}, {"silence": 0.8}, 0, 0, False, True
    if i <= 15:
        v = {"instruction": 0.9, "actor_staff_facilitation": 0.9, "actor_participant": 0.1}
        return v, {"speech": 0.7}, 2, 0, True, False
    if i <= 18:
        v = {"lyricism_spoken_word": 0.9, "performance": 0.9, "actor_participant": 0.9}
        return v, {"speech": 0.8}, 1, 2, True, False
    return {"audience_watching": 0.9, "participation": 0.6}, {"applause": 0.8}, 6, 0, False, False


def _segments(vid, n):
    starts = [i * 6.0 for i in range(n)]
    return [(f"{vid}_{round(s * 1000)}", s, s + 8.0 if i < n - 1 else s + 6.0) for i, s in enumerate(starts)]


def _run(stage, rid, videos, segments, preds=None, moments=None, model=None):
    write_table("videos", videos, rid)
    write_table("segments", segments, rid)
    if preds is not None:
        write_table("predictions", preds, rid)
    if moments is not None:
        write_table("moments", moments, rid)
    write_table("runs", [{"run_id": rid, "stage": stage, "model": model, "config_hash": "0" * 12, "hardware": "mock",
                          "n_segments": len(segments), "runtime_s": 1.0, "peak_mem_gb": None, "started_at": T0,
                          "git_commit": None}], rid)


def build_world(tmp_path, monkeypatch, n_extra_eval=0, n_segments=20, n_labels=None, overrides=None):
    """Write the mock world. n_extra_eval / n_segments / n_labels scale it (SC-007). Returns ids for assertions."""
    overrides = overrides or {}
    labels = load_labels()
    visual = [lab["name"] for lab in model_labels(labels, {"activity", "category", "actor", "null"})]
    thresholds = {lab["name"]: 0.5 for lab in model_labels(labels, {"activity", "category", "actor"})}
    config.THRESHOLDS_LOCAL.write_text(yaml.safe_dump({"label_thresholds": {SIGLIP: thresholds}}))
    pipeline = load_pipeline()

    videos_spec = list(VIDEOS) + [(f"e{i + 3:02d}", "eval", "showcase") for i in range(n_extra_eval)]
    vrows, base, preds, audio, people, expr, transcript = [], [], [], [], [], [], []
    for vid, split, phase in videos_spec:
        segs = _segments(vid, n_segments)
        vrows.append({"video_id": vid, "filename": f"mock/{SECRET if vid == 'e01' else vid}.mp4",
                      "duration_s": segs[-1][2], "fps": 30.0, "orientation": "landscape", "kind": "live",
                      "phase": phase, "cohort": None, "split": split})
        for i, (sid, start, end) in enumerate(segs):
            p = i % 20
            v, a, persons, faces, speech, idle = _pattern(p)
            v, a = dict(v), dict(a)
            if vid == "e01" and p == 4:  # planted: visual singing without singing audio
                a = {"singing_voice": 0.1, "music": 0.1}
            if vid == "e01" and 5 <= p <= 9:
                v["dance"] = 0.55
            if vid == "e02" and 10 <= p <= 14:  # planted staff stretch (model only)
                v, a, persons, faces, speech, idle = ({"actor_staff_facilitation": 0.9, "actor_participant": 0.1},
                                                      {"speech": 0.7}, 1, 0, True, False)
            seg = {"segment_id": sid, "video_id": vid, "start_s": start, "end_s": end, "person_count": None,
                   "faces_detected": None, "has_speech": None, "transcript_chars": None, "is_idle": idle,
                   "idle_score": 0.9 if idle else 0.05, "visual_change": 0.0 if idle else 0.2}
            base.append(seg)
            preds += [{"segment_id": sid, "label": lab, "label_group": _group(labels, lab), "score": v.get(lab, 0.05),
                       "model": SIGLIP, "run_id": "predict-20261024T000000-mock"} for lab in visual]
            audio += [{"segment_id": sid, "label": lab, "label_group": "audio", "score": a.get(lab, 0.02),
                       "model": AST, "run_id": "audio-20261024T000000-mock"} for lab in AUDIO]
            people.append({**seg, "is_idle": None, "idle_score": None, "visual_change": None, "person_count": persons})
            expr.append({**seg, "is_idle": None, "idle_score": None, "visual_change": None, "faces_detected": faces})
            if faces:
                top = "positive" if vid == "e02" else "neutral"
                expr_scores = {lab: (0.6 if lab == top else 0.1) for lab in EXPR}
                preds_expr = [{"segment_id": sid, "label": lab, "label_group": "expression", "score": s,
                               "model": ENET, "run_id": "expression-20261024T000000-mock"}
                              for lab, s in expr_scores.items()]
                overrides.setdefault("_expr", []).extend(preds_expr)
            transcript.append({**seg, "is_idle": None, "idle_score": None, "visual_change": None,
                               "has_speech": speech, "transcript_chars": 40 if speech else 0})

    rules = rules_rows(base, preds, pipeline, labels, SIGLIP, "predict-20261024T000000-mock")
    moments = []
    for vid, *_ in videos_spec:
        moments += merge_moments([s for s in base if s["video_id"] == vid], preds + rules, pipeline,
                                 "predict-20261024T000000-mock", window_s=8, model=SIGLIP)
    skip_stages = set(overrides.get("skip_stages", ()))
    _run("predict", "predict-20261024T000000-mock", vrows, base, preds + rules, moments, SIGLIP)
    stage_rows = {"audio": (people_like(base), audio, AST), "people": (people, None, "mock-yolo"),
                  "expression": (expr, overrides.pop("_expr", []), ENET),
                  "transcript": (transcript, None, "mock-whisper")}
    official = {"predict": ["predict-20261024T000000-mock"]}
    for stage, (segs, stage_preds, model) in stage_rows.items():
        rid = f"{stage}-20261024T000000-mock"
        keep = [v for v in vrows if (stage, v["video_id"]) not in skip_stages]
        kept = {v["video_id"] for v in keep}
        _run(stage, rid, keep, [s for s in segs if s["video_id"] in kept],
             None if stage_preds is None else [p for p in stage_preds if p["segment_id"].rsplit("_", 1)[0] in kept],
             model=model)
        official[stage] = [rid]
    (tmp_path / "out/runs/transcript-20261024T000000-mock/transcript.json").write_text(
        json.dumps({"video_id": "e01", "segments": [{"text": SENTINEL}]}))
    path = tmp_path / "official.yaml"
    path.write_text(yaml.safe_dump(official))
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)

    seg_by_id = {s["segment_id"]: s for s in base}
    ids = {vid: [sid for sid, *_ in _segments(vid, n_segments)] for vid, *_ in videos_spec}
    main = ids["e01"][:20] + ids["e02"][:10] + ids["d01"][:3]
    for vid in [v for v, *_ in videos_spec[4:]]:
        main += ids[vid]
    if n_labels:
        eval_ids = [sid for vid, split, _ in videos_spec if split == "eval" for sid in ids[vid]]
        main = eval_ids[:n_labels] + ids["d01"][:3]
    topup = ids["e02"][15:18]
    _queue("main", [seg_by_id[s] for s in main])
    _queue("topup", [seg_by_id[s] for s in topup], target="dance")
    rows = []
    for sid in main + topup:
        rows += _blind(seg_by_id[sid], labels, pipeline)
    append_reviews(rows)
    # e01 seg 0: `dance` was added, then unticked in a later save (latest state: rejected)
    first = blind_rows(seg_by_id[ids["e01"][0]], "singing", ["performance", "actor_participant", "dance"], False,
                       "mock-staff-a", labels, pipeline)
    append_reviews(first)
    previous = {r["label"]: r for r in first}
    append_reviews(blind_rows(seg_by_id[ids["e01"][0]], "singing", ["performance", "actor_participant",
                                                                     "solo_on_stage"], False, "mock-staff-a", labels,
                              pipeline, previous=previous))
    return {"pipeline": pipeline, "labels": labels, "ids": ids, "main": main, "topup": topup,
            "fusion_segment": ids["e01"][4], "skip_segment": ids["e02"][9], "two_categories": ids["e01"][5],
            "no_category": ids["e01"][6], "wrong_primary": ids["e02"][3], "staff_stretch": ids["e02"][10:15],
            "idle": ids["e01"][10:13], "instruction": ids["e01"][13:16]}


def people_like(base):
    """Segment rows for the audio stage (no signal columns of its own)."""
    return [{**s, "is_idle": None, "idle_score": None, "visual_change": None} for s in base]


def _group(labels, name):
    return next(lab["group"] for lab in labels["labels"] if lab["name"] == name)


def _queue(name, segs, target=None):
    append_blind_queue([{"queue": name, "position": i, "segment_id": s["segment_id"], "video_id": s["video_id"],
                         "start_s": s["start_s"], "end_s": s["end_s"], "target_label": target, "seed": 2026,
                         "created_at": T0} for i, s in enumerate(segs)])


def _blind(seg, labels, pipeline):
    """The labeler's answer for one segment (see the module docstring)."""
    vid, p = seg["video_id"], round(seg["start_s"] / 6) % 20
    if vid == "e02" and p == 9:
        return [skip_row(seg, "mock-staff-a", labels, pipeline)]
    if vid == "e01" and p == 0:
        return []  # written separately (add, then reject)
    if p <= 4:
        primary = "dance" if (vid == "e02" and p == 3) else "singing"
        chosen = ["performance", "actor_participant", "solo_on_stage" if p <= 1 else "group_performance"]
        if vid == "e02":
            chosen.append("voice_projection")
        return blind_rows(seg, primary, chosen, False, "mock-staff-a", labels, pipeline)
    if p <= 9:
        chosen = ["performance", "actor_participant", "group_performance"]
        if vid == "e01" and p == 5:
            chosen.append("participation")
        if vid == "e01" and p == 6:
            chosen = ["actor_participant"]
        return blind_rows(seg, "dance", chosen, False, "mock-staff-a", labels, pipeline)
    if p <= 12:
        return blind_rows(seg, None, [], True, "mock-staff-a", labels, pipeline)
    if p <= 15:
        return blind_rows(seg, None, ["instruction", "actor_staff_facilitation"], True, "mock-staff-a", labels,
                          pipeline)
    if p <= 18:
        return blind_rows(seg, "lyricism_spoken_word", ["performance", "actor_participant", "voice_projection"], False,
                          "mock-staff-a", labels, pipeline)
    return blind_rows(seg, "audience_watching", ["participation", "actor_participant"], False, "mock-staff-a",
                      labels, pipeline)
