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
