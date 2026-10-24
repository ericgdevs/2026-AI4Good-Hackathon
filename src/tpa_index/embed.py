"""Image/text embedders. Scoring code depends only on the Embedder protocol, so tests can use a fake."""
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str  # recorded in predictions.model and runs.model
    logit_scale: float  # already exp()'d
    logit_bias: float

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...  # (n, d) float32, L2-normalized

    def embed_images(self, images: np.ndarray) -> np.ndarray: ...  # (n, S, S, 3) uint8 -> (n, d) float32, L2-normalized


def _features(out):
    # transformers may return a tensor or an output object, depending on version
    return out if hasattr(out, "float") else out.pooler_output


class SiglipEmbedder:
    """SigLIP/SigLIP2 from the local Hugging Face cache only (never downloads)."""

    def __init__(self, model_id: str, device: str, batch_images: int = 256):
        import torch
        from transformers import AutoModel, AutoProcessor

        from tpa_index.config import ConfigError

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_id, local_files_only=True, dtype=dtype).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            raise ConfigError(f"model not found locally: {model_id}") from None
        self.torch, self.device, self.dtype, self.batch_images = torch, device, dtype, batch_images
        self.name = model_id.rsplit("/", 1)[-1]
        self.logit_scale = self.model.logit_scale.detach().float().exp().item()
        self.logit_bias = self.model.logit_bias.detach().float().item()

    def _unit(self, t) -> np.ndarray:
        t = t.float()
        return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # SigLIP was trained with lowercase text padded to 64 tokens; other padding shifts the scores
        inputs = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self._unit(_features(self.model.get_text_features(**inputs)))

    def embed_images(self, images: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_images):
            inputs = self.processor(images=list(images[i:i + self.batch_images]), return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device, self.dtype)
            with self.torch.inference_mode():
                out.append(self._unit(_features(self.model.get_image_features(pixel_values=pixels))))
        return np.concatenate(out)
