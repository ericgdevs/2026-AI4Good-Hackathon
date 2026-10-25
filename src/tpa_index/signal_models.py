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


def cpu_threads() -> int:
    """CPU threads for OpenCV/onnxruntime/torch: TPA_THREADS, else the CPU count capped by the Slurm allocation
    (library defaults follow the whole node and oversubscribe an 8-CPU job)."""
    import os

    if os.environ.get("TPA_THREADS"):
        return max(1, int(os.environ["TPA_THREADS"]))
    n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, min(n, int(slurm))) if slurm else n


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
        import soundfile as sf

        # an array, not the path: faster-whisper 1.2.1's own decoder passes an argument PyAV 19 rejects
        audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        segments, info = self.model.transcribe(audio, vad_filter=True, word_timestamps=True,
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


class PersonCounter(Protocol):
    name: str

    def count(self, frames: np.ndarray) -> list[int]: ...  # (N,S,S,3) RGB -> persons per frame


class ExpressionScorer(Protocol):
    name: str
    labels: list[str]

    def faces(self, frame: np.ndarray) -> list[np.ndarray]: ...  # usable in-memory RGB crops

    def score(self, crops: list[np.ndarray]) -> np.ndarray: ...  # (n_faces, len(labels)), mapped sums


class YoloPersonCounter:
    """YOLO11, person class only, detection only (no tracking). Runs keep only the count per frame; boxes reach the
    Process tab's in-memory preview, never a file."""

    def __init__(self, weights_entry: dict, device: str, conf: float, min_box_height_frac: float):
        import os

        from tpa_index.config import pinned_weight

        path = pinned_weight(weights_entry)  # first: a missing file never reaches ultralytics (it would download)
        os.environ["YOLO_OFFLINE"] = "1"
        from ultralytics import YOLO

        import torch

        torch.set_num_threads(cpu_threads())  # pre/post-processing runs on the CPU
        self.model, self.device, self.conf = YOLO(str(path)), device, conf
        self.min_h = min_box_height_frac
        self.name = Path(weights_entry["file"]).stem

    def boxes(self, frames: np.ndarray) -> list[list[tuple[float, float, float, float]]]:
        """Counted person boxes per frame as (x0, y0, x1, y1) in 0-1 of the frame; small/distant people are dropped."""
        results = self.model.predict(list(frames), classes=[0], conf=self.conf, verbose=False, device=self.device)
        out = []
        for r in results:
            xywhn = r.boxes.xywhn
            xywhn = xywhn.cpu().numpy() if hasattr(xywhn, "cpu") else xywhn  # a tensor on the model's device
            x, y, w, h = np.asarray(xywhn, np.float32).reshape(-1, 4).T
            out.append([(float(a - c / 2), float(b - d / 2), float(a + c / 2), float(b + d / 2))
                        for a, b, c, d in zip(x, y, w, h) if d >= self.min_h])
        return out

    def count(self, frames: np.ndarray) -> list[int]:
        return [len(b) for b in self.boxes(frames)]


def usable_faces(detections, frame_shape, face_score: float, min_face_px: int) -> list[tuple[int, int, int, int]]:
    """YuNet rows [x, y, w, h, 10 landmark values, score] -> boxes clipped to the frame, kept when the score and
    the shorter side are high enough. Landmarks are never returned."""
    if detections is None:
        return []
    h_img, w_img = frame_shape[:2]
    boxes = []
    for row in detections:
        if row[14] < face_score:
            continue
        x0, y0 = max(int(row[0]), 0), max(int(row[1]), 0)
        x1, y1 = min(int(row[0] + row[2]), w_img), min(int(row[1] + row[3]), h_img)
        if min(x1 - x0, y1 - y0) >= min_face_px:
            boxes.append((x0, y0, x1, y1))
    return boxes


def to_source_pixels(detections, scale: float):
    """YuNet rows found on a frame resized by `scale` -> the same rows with x, y, w, h in source pixels."""
    if detections is None or scale == 1.0:
        return detections
    out = np.array(detections, np.float32, copy=True)
    out[:, :4] /= scale
    return out


def yunet_file(entry: dict) -> str:
    """Local path of the pinned YuNet ONNX file, checked against its sha256. Never downloads."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from tpa_index.config import check_sha256

    try:
        path = hf_hub_download(entry["repo"], entry["file"], revision=entry["revision"], local_files_only=True)
    except LocalEntryNotFoundError:
        raise ConfigError(f"model not found locally: {entry['repo']}") from None
    check_sha256(Path(path), entry)
    return path


def detect_faces(detector, frame: np.ndarray, max_side: int):
    """YuNet rows for an RGB frame, in source pixels. Detection runs on a copy whose longer side is at most
    max_side; landmarks are in the rows but callers keep boxes only."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = frame if scale == 1.0 else cv2.resize(
        frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, detections = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return to_source_pixels(detections, scale)


class FaceExpression:
    """YuNet face detection, then EmotiEffLib (ONNX, CPU) on in-memory crops. Classes mapped to our expression
    labels in config (sum rule, no renormalization); valence/arousal outputs are dropped."""

    def __init__(self, detector_entry: dict, model_entry: dict, mapping: dict[str, list[str]], face_score: float,
                 min_face_px: int, detect_max_side: int = 1280):
        import cv2

        from tpa_index.config import pinned_weight

        path = pinned_weight(model_entry)  # before importing the library: it downloads missing files
        yunet = yunet_file(detector_entry)
        import emotiefflib.facial_analysis as fa

        import types

        import onnxruntime as ort

        threads = cpu_threads()
        cv2.setNumThreads(threads)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        fa.get_model_path_onnx = lambda name: str(path)  # shortcut: library has no path argument; drop when it gets one
        # the library builds its session with no options; swap only its module-level `ort` name while it does
        fa.ort = types.SimpleNamespace(
            set_default_logger_severity=ort.set_default_logger_severity,
            InferenceSession=lambda model, providers=None: ort.InferenceSession(model, sess_options=opts,
                                                                                providers=providers))
        try:
            self.recognizer = fa.EmotiEffLibRecognizerOnnx(model_entry["name"])
        finally:
            fa.ort = ort
        classes = list(self.recognizer.idx_to_emotion_class.values())
        missing = [c for cs in mapping.values() for c in cs if c not in classes]
        if missing:
            raise ConfigError(f"signals.expression.mapping: class {missing[0]!r} not in model classes")
        self.columns = {label: [classes.index(c) for c in cs] for label, cs in mapping.items()}
        self.n_classes = len(classes)
        self.detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320), face_score)
        self.cv2, self.face_score, self.min_face_px, self.detect_max_side = cv2, face_score, min_face_px, detect_max_side
        self.labels, self.name = list(mapping), model_entry["name"]

    def faces(self, frame: np.ndarray) -> list[np.ndarray]:
        """Detect on a copy whose longer side is at most detect_max_side, map boxes back to source pixels, then apply
        the score and min_face_px rules in source pixels and cut crops from the original frame."""
        detections = detect_faces(self.detector, frame, self.detect_max_side)
        boxes = usable_faces(detections, frame.shape, self.face_score, self.min_face_px)
        del detections  # boxes only; landmarks dropped here
        return [frame[y0:y1, x0:x1].copy() for x0, y0, x1, y1 in boxes]

    def score(self, crops: list[np.ndarray]) -> np.ndarray:
        _, probs = self.recognizer.predict_emotions(crops, logits=False)
        probs = np.asarray(probs)[:, :self.n_classes]  # trailing valence/arousal columns are never kept
        return np.stack([probs[:, cols].sum(axis=1) for cols in self.columns.values()], axis=1).astype(np.float32)
