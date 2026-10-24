"""004 signal models with real weights from the local cache (never downloaded). One small model per test, on CPU,
released afterwards (the Mac also runs the labeling app). Synthetic audio only. Run: uv run pytest -m slow."""
import gc

import numpy as np
import pytest

from tpa_index.config import ConfigError

pytestmark = pytest.mark.slow
AUDIO_LABELS = ["singing_voice", "music", "speech", "applause", "silence"]


def _release(*objs):
    for o in objs:
        del o
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _whisper_or_skip():
    from tpa_index.signal_models import WhisperTranscriber

    for size in ("tiny", "base"):
        try:
            return WhisperTranscriber(size, "cpu", "int8")
        except ConfigError:
            continue
    pytest.skip("faster-whisper tiny/base not in the local HF cache: fetch Systran/faster-whisper-base "
                "(and -tiny) with scripts/download_weights.sh")


def test_whisper_on_tone_finds_no_words(tmp_path, monkeypatch):
    import soundfile as sf

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    t = np.arange(3 * 16000) / 16000
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, subtype="PCM_16")
    model = _whisper_or_skip()
    try:
        segments = model.transcribe(wav)
        n_words = sum(len(s["words"]) for s in segments)  # counts only: never print or assert on text
        assert n_words == 0
    finally:
        _release(model)


def test_ast_mapping_resolves_and_scores_silence(monkeypatch):
    from tpa_index.config import load_pipeline
    from tpa_index.signal_models import AstTagger

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_pipeline()
    try:
        model = AstTagger(cfg["models"]["audio"]["mac"], "cpu", cfg["signals"]["audio"]["mapping"])
    except ConfigError as e:
        if "not found locally" in str(e):
            pytest.skip("AST not in the local HF cache: fetch MIT/ast-finetuned-audioset-10-10-0.4593")
        raise  # a mapping name missing from the model is a real failure
    try:
        assert model.labels == AUDIO_LABELS and model.input_samples == 163840
        scores = model.score([np.zeros(model.input_samples, np.float32)] * 2)
        assert scores.shape == (2, 5) and ((scores >= 0) & (scores <= 1)).all()
    finally:
        _release(model)
