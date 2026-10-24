"""Thin wrappers around the 004 signal models. Weights load from local files only (never downloaded); every load
problem is a ConfigError raised before any video is processed. Orchestration depends only on the Protocols, so
tests use fakes (tests/fakes.py)."""
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from tpa_index.config import ConfigError


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav_path: Path) -> list[dict]: ...  # [{"start","end","text","words":[{"start","end","word"}]}]


class AudioTagger(Protocol):
    name: str
    labels: list[str]
    input_samples: int

    def score(self, clips: list[np.ndarray]) -> np.ndarray: ...  # (n, len(labels)) in 0-1


_LIB = re.compile(r"lib[\w.+-]*\.so[\d.]*|[\w.+-]+\.dll")


class WhisperTranscriber:
    """faster-whisper (CTranslate2): whole-file transcription with VAD and word timestamps."""

    def __init__(self, size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.name, self.language = f"faster-whisper-{size}", None
        try:
            self.model = WhisperModel(size, device=device, compute_type=compute_type, local_files_only=True)
            if device == "cuda":  # CTranslate2 loads CUDA libraries lazily: fail here, before any video
                list(self.model.transcribe(np.zeros(16000, np.float32), vad_filter=False)[0])
        except LocalEntryNotFoundError:
            raise ConfigError(f"model not found locally: {self.name}") from None
        except Exception as e:  # noqa: BLE001
            if device != "cuda":
                raise ConfigError(f"transcript model could not load on {device}: {type(e).__name__}") from None
            lib = _LIB.search(str(e))
            raise ConfigError(f"transcript model could not load on cuda: missing {lib.group(0) if lib else type(e).__name__}"
                              "; set models.transcript.hpc_device: cpu to run on CPU") from None

    def transcribe(self, wav_path: Path) -> list[dict]:
        segments, info = self.model.transcribe(str(wav_path), vad_filter=True, word_timestamps=True,
                                               condition_on_previous_text=False)
        out = [{"start": s.start, "end": s.end, "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words or []]}
               for s in segments]
        self.language = info.language
        return out


class AstTagger:
    """AST finetuned on AudioSet; AudioSet classes mapped to our audio labels in config (max rule)."""

    def __init__(self, model_id: str, device: str, mapping: dict[str, list[str]]):
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        try:
            self.model = ASTForAudioClassification.from_pretrained(model_id, local_files_only=True).to(device).eval()
            self.extractor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            raise ConfigError(f"model not found locally: {model_id}") from None
        index = {name: int(i) for i, name in self.model.config.id2label.items()}
        missing = [c for classes in mapping.values() for c in classes if c not in index]
        if missing:
            raise ConfigError(f"signals.audio.mapping: class {missing[0]!r} not in model labels")
        self.class_index = {label: [index[c] for c in classes] for label, classes in mapping.items()}
        self.labels, self.torch, self.device = list(mapping), torch, device
        self.name = model_id.rsplit("/", 1)[-1]
        self.rate = self.extractor.sampling_rate
        self.input_samples = int(self.extractor.max_length * self.rate // 100)  # 10 ms hop: 1024 frames = 10.24 s

    def score(self, clips: list[np.ndarray]) -> np.ndarray:
        from tpa_index.signals import map_scores

        inputs = self.extractor(clips, sampling_rate=self.rate, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(input_values=inputs["input_values"].to(self.device)).logits
        return map_scores(self.torch.sigmoid(logits.float()).cpu().numpy(), self.class_index)
