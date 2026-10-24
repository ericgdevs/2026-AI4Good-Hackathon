"""004 privacy and identity guarantees (SC-001 to SC-005, SC-013). Fakes and synthetic clips only.
Leak checks compute a boolean first, so a failing assertion can never echo transcript text."""
import os

import pytest

from tests.fakes import FakeAudioTagger, FakeTranscriber
from tests.test_cli import _predict_yaml, signals_cli
from tpa_index import schema, signals

SENTINEL = "ZQXJ-SENTINEL-7781"
AUDIO_LABELS = ["singing_voice", "music", "speech", "applause", "silence"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy", ".webp", ".tif", ".tiff"}


def _contains(path, needle: str) -> bool:
    return needle.encode() in path.read_bytes()


def test_transcript_text_never_leaks(monkeypatch, clip_videos, pipeline, tmp_path, caplog, capsys):
    """SC-003: success on t01, a failure carrying the sentinel in its message on t03."""
    import logging

    caplog.set_level(logging.DEBUG)
    models = {"transcript": FakeTranscriber(sentinel=SENTINEL, fail_for={"t03"}, raise_with=f"boom {SENTINEL}")}
    cli, sel = signals_cli(monkeypatch, models, clip_videos, pipeline, tmp_path)
    code = cli.main(["signals", "--stages", "transcript", *sel])
    out, err = capsys.readouterr()
    assert code == 1  # t03 failed
    leaked = {"logs": SENTINEL in caplog.text, "stdout": SENTINEL in out, "stderr": SENTINEL in err,
              "parquet": any(_contains(p, SENTINEL) for p in (tmp_path / "out").rglob("*.parquet"))}
    assert not any(leaked.values()), f"sentinel leaked into: {[k for k, v in leaked.items() if v]}"
    holders = sorted(p.parent.name.split("-")[0] for p in (tmp_path / "out").rglob("*") if p.is_file()
                     and _contains(p, SENTINEL))
    assert holders == ["transcript"]  # only the successful run's transcript.json
    assert [p.name for p in (tmp_path / "out").rglob("*") if p.is_file() and _contains(p, SENTINEL)] == \
        ["transcript.json"]


def test_transcript_error_carries_no_text(clip_videos, pipeline, tmp_path, monkeypatch):
    """SC-003: the raised error, its context and its traceback hold only the original type name."""
    import traceback

    from tests.test_signals import official_clip_runs

    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    model = FakeTranscriber(fail_for={"t01"}, raise_with=SENTINEL)
    wav = tmp_path / "tpa-audio-1-t01.wav"
    with pytest.raises(signals.TranscriptError) as err:
        signals._transcript_stage(model, wav, signals.official_video("t01"), signals.official_segments("t01"),
                                  "transcript-x", None)
    text = "".join(traceback.format_exception(err.value)) + repr(err.value)
    leaked = SENTINEL in text
    assert not leaked, "sentinel leaked into the exception"
    assert str(err.value) == "RuntimeError" and err.value.__context__ is None and err.value.__cause__ is None


def _stray_files(*roots):
    found = []
    for root in roots:
        found += [p.name for p in root.rglob("*") if p.is_file()
                  and (p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".wav")]
    return found


def test_no_temp_or_image_files_survive(monkeypatch, clip_videos, pipeline, tmp_path):
    """SC-002: after success, failure and a simulated kill, no temp WAV and no image file is left."""
    import psutil

    models = {"transcript": FakeTranscriber(), "audio": FakeAudioTagger(AUDIO_LABELS)}
    cli, sel = signals_cli(monkeypatch, models, clip_videos, pipeline, tmp_path)
    scratch, out = tmp_path / "scratch", tmp_path / "out"
    assert cli.main(["signals", "--stages", "transcript,audio", *sel]) == 0
    assert _stray_files(scratch, out) == []

    models["audio"].fail_on_call = models["audio"].calls + 1  # next audio call raises
    assert cli.main(["signals", "--stages", "transcript,audio", "--force", *sel]) == 1
    assert _stray_files(scratch, out) == []

    dead = 999_999
    while psutil.pid_exists(dead):
        dead += 1
    (scratch / f"tpa-audio-{dead}-t01.wav").write_bytes(b"RIFF")  # left by a killed job
    assert cli.main(["signals", "--stages", "audio", "--force", *sel]) == 0
    assert _stray_files(scratch, out) == []


def test_bad_audio_mapping_name_exits_before_any_video(monkeypatch, clip_videos, pipeline, tmp_path, capsys):
    """SC-013 (audio half): a configured AudioSet name the model lacks stops the command at startup."""
    import transformers

    from tests.test_signals import official_clip_runs
    from tpa_index import cli

    class FakeAst:
        config = type("C", (), {"id2label": {i: n for i, n in enumerate(
            ["Singing", "Choir", "Male singing", "Female singing", "Child singing", "Rapping", "Humming", "Music",
             "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
             "Conversation", "Narration, monologue", "Applause", "Clapping", "Cheering", "Silence"])}})()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeExtractor:
        sampling_rate, max_length = 16000, 1024

    monkeypatch.setattr(transformers.ASTForAudioClassification, "from_pretrained", lambda *a, **k: FakeAst())
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", lambda *a, **k: FakeExtractor())
    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    code = cli.main(["signals", "--stages", "audio", "--videos", str(_predict_yaml(tmp_path)), "--video-id", "t01"])
    assert code == 3
    assert "class 'Chant' not in model labels" in capsys.readouterr().err
    assert {r["stage"] for r in schema.read_table("runs").to_pylist()} == {"segment"}
    assert not [d for d in (tmp_path / "out/runs").iterdir() if d.name.startswith("audio-")]


def test_scratch_never_in_repo(monkeypatch):
    from tpa_index.paths import GuardError, repo_root

    monkeypatch.setenv("TPA_SCRATCH_DIR", os.fspath(repo_root()))
    with pytest.raises(GuardError):
        signals.scratch_dir()


# ---- job B: people + expression ------------------------------------------------------------------------------------
EXPR_LABELS = ["positive", "neutral", "negative", "upset"]


class TinyFaces:
    """A detector whose only faces are 30 px (below min_face_px 48), filtered by the real size rule."""

    name, labels = "tiny-faces", EXPR_LABELS

    def faces(self, frame):
        import numpy as np

        from tpa_index.signal_models import usable_faces

        dets = np.array([[5, 5, 30, 30, *range(10), 0.99]], np.float32)
        return [frame[y0:y1, x0:x1].copy() for x0, y0, x1, y1 in usable_faces(dets, frame.shape, 0.8, 48)]

    def score(self, crops):
        raise AssertionError("never called without faces")


@pytest.mark.parametrize("scorer", ["no_faces", "too_small"])
def test_zero_faces_give_no_expression_rows(scorer, clip_videos, pipeline, labels, tmp_path, monkeypatch):
    """SC-004: 0 usable faces -> faces_detected 0 and no expression rows (never a default 'neutral')."""
    from tests.fakes import FakeExpressionScorer
    from tests.test_signals import official_clip_runs

    official_clip_runs([clip_videos["t04"]], pipeline, tmp_path, monkeypatch)
    model = FakeExpressionScorer(EXPR_LABELS, faces_per_frame=0) if scorer == "no_faces" else TinyFaces()
    results = signals.run_signals([clip_videos["t04"]], ["expression"], {"expression": model}, pipeline, labels)
    run_id = results[("t04", "expression")].split(":")[1]
    segs = schema.read_table("segments", run_ids=[run_id]).to_pylist()
    assert segs and {s["faces_detected"] for s in segs} == {0}
    assert schema.read_table("predictions", run_ids=[run_id]).num_rows == 0


def test_no_identity_columns_or_files(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    """SC-001: stage runs hold only contract tables (+ transcript.json); no identity column anywhere."""
    import re

    import pyarrow.parquet as pq

    from tests.fakes import FakeExpressionScorer, FakePersonCounter
    from tests.test_signals import official_clip_runs

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    models = {"transcript": FakeTranscriber(), "audio": FakeAudioTagger(AUDIO_LABELS),
              "people": FakePersonCounter(), "expression": FakeExpressionScorer(EXPR_LABELS, faces_per_frame=3)}
    results = signals.run_signals([clip_videos["t01"]], list(models), models, pipeline, labels)
    assert all(v.startswith("done:") for v in results.values())
    identity = re.compile(r"(?i)(embed|track|landmark|bbox|box|crop|face_id|person_id|identity)")
    assert not [c for s in schema.SCHEMAS.values() for c in s.names if identity.search(c)]
    tables = {"videos.parquet", "segments.parquet", "predictions.parquet", "runs.parquet"}
    for (_, stage), status in results.items():
        folder = tmp_path / "out/runs" / status.split(":")[1]
        files = {p.name for p in folder.iterdir()}
        assert files <= tables | ({"transcript.json"} if stage == "transcript" else set()), (stage, files)
        for f in folder.glob("*.parquet"):
            assert not [c for c in pq.read_schema(f).names if identity.search(c)]
    for stage in ("audio", "expression"):
        preds = schema.read_table("predictions", run_ids=[results[("t01", stage)].split(":")[1]]).to_pylist()
        per_segment = {}
        for p in preds:
            per_segment[p["segment_id"]] = per_segment.get(p["segment_id"], 0) + 1
        n_labels = len(AUDIO_LABELS if stage == "audio" else EXPR_LABELS)
        assert set(per_segment.values()) == {n_labels}  # one row per label, never per face or person


def test_bad_expression_mapping_class_exits_before_any_video(monkeypatch, clip_videos, pipeline, tmp_path, capsys):
    """SC-013 (expression half): a mapped class the model lacks stops the command at startup."""
    import copy

    import emotiefflib.facial_analysis as fa
    import huggingface_hub

    from tests.test_signals import official_clip_runs
    from tpa_index import cli, config

    class FakeRecognizer:
        def __init__(self, name):
            self.idx_to_emotion_class = dict(enumerate(
                ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]))

    bad = copy.deepcopy(pipeline)
    bad["signals"]["expression"]["mapping"]["positive"] = ["Happyness"]
    monkeypatch.setattr(cli, "load_pipeline", lambda: bad)
    monkeypatch.setattr(config, "pinned_weight", lambda entry: tmp_path / entry["file"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(tmp_path / "yunet.onnx"))
    monkeypatch.setattr(fa, "EmotiEffLibRecognizerOnnx", FakeRecognizer)
    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    code = cli.main(["signals", "--stages", "expression", "--videos", str(_predict_yaml(tmp_path)),
                     "--video-id", "t01"])
    assert code == 3
    assert "class 'Happyness' not in model classes" in capsys.readouterr().err
    assert not [d for d in (tmp_path / "out/runs").iterdir() if d.name.startswith("expression-")]


def test_no_image_files_after_job_b(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    """SC-002 for job B: frames and crops never reach disk, also when expression fails mid-pass."""
    from tests.fakes import FakeExpressionScorer, FakePersonCounter
    from tests.test_signals import official_clip_runs

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    for fail in (None, 2):
        models = {"people": FakePersonCounter(),
                  "expression": FakeExpressionScorer(EXPR_LABELS, faces_per_frame=2, fail_on_call=fail)}
        signals.run_signals([clip_videos["t01"]], list(models), models, pipeline, labels, force=True)
        assert _stray_files(tmp_path / "scratch", tmp_path / "out") == []


def test_official_002_runs_unchanged(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    """SC-005: official segment, predict and moments runs and the existing official_runs.yaml sections are
    byte-identical after all four stages run; the default official reads return the same rows."""
    import hashlib

    from tests.fakes import FakeExpressionScorer, FakePersonCounter
    from tests.test_schema import row
    from tests.test_signals import official_clip_runs

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    vids = [clip_videos["t01"], clip_videos["t03"]]
    (seg_a, seg_b) = official_clip_runs(vids, pipeline, tmp_path, monkeypatch)
    official = {"segment": [seg_a, seg_b], "predict": [], "moments": []}
    for seg_run in (seg_a, seg_b):
        vid = seg_run.rsplit("-", 1)[1]
        for stage in ("predict", "moments"):
            rid = f"{stage}-20261024T000000-{vid}"
            segs = schema.read_table("segments", run_ids=[seg_run]).to_pylist()
            schema.write_table("videos", schema.read_table("videos", run_ids=[seg_run]), rid)
            schema.write_table("segments", segs, rid)
            schema.write_table("predictions", [row("predictions", segment_id=s["segment_id"], model="m",
                                                   run_id=rid) for s in segs], rid)
            schema.write_table("moments", [row("moments", moment_id=f"{vid}_0_singing", video_id=vid,
                                               run_id=rid)], rid)
            schema.write_table("runs", [row("runs", run_id=rid, stage=stage)], rid)
            official[stage].append(rid)
    yaml_path = tmp_path / "official.yaml"
    yaml_path.write_text("".join(f"{k}:\n" + "".join(f"  - {r}\n" for r in v) for k, v in official.items()))
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", yaml_path)

    def hashes():
        return {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for rid in sum(official.values(), []) for p in (tmp_path / "out/runs" / rid).iterdir()}

    before_files, before_yaml = hashes(), yaml_path.read_bytes()
    before_reads = {t: schema.read_official(t).to_pylist() for t in ("segments", "predictions", "moments")}
    models = {"transcript": FakeTranscriber(), "audio": FakeAudioTagger(AUDIO_LABELS),
              "people": FakePersonCounter(), "expression": FakeExpressionScorer(EXPR_LABELS)}
    results = signals.run_signals(vids, list(models), models, pipeline, labels)
    assert all(v.startswith("done:") for v in results.values())
    with open(yaml_path, "a") as f:  # how the team lists the new runs: append sections, never edit existing ones
        f.write("# 004 signal stages (one run per video per stage)\n")
        for stage in ("transcript", "audio", "people", "expression"):
            f.write(f"{stage}:\n" + "".join(f"  - {status.split(':')[1]}\n"
                                            for (v, s), status in sorted(results.items()) if s == stage))
    assert hashes() == before_files
    assert yaml_path.read_bytes().startswith(before_yaml)
    assert {t: schema.read_official(t).to_pylist() for t in before_reads} == before_reads
    assert schema.read_signals().num_rows == len(before_reads["segments"])
