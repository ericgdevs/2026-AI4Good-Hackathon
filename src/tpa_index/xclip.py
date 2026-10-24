"""Optional motion model (X-CLIP) for a few motion labels; imported only when models.motion.enabled."""
import numpy as np


class XClipScorer:
    """Scores are a softmax over [label prompts..., other_prompt]: relative to each other, not independent."""

    def __init__(self, model_id: str, device: str, labels: list[dict], other_prompt: str, frames_per_window: int = 8):
        import torch
        from transformers import AutoProcessor, XCLIPModel

        from tpa_index.config import ConfigError

        try:
            self.model = XCLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            raise ConfigError(f"model not found locally: {model_id}") from None
        self.torch, self.device = torch, device
        self.name = model_id.rsplit("/", 1)[-1]
        self.labels = [lab["name"] for lab in labels]
        self.prompts = [lab["prompts"][0] for lab in labels] + [other_prompt]
        self.frames_per_window = frames_per_window  # X-CLIP's frame-position embedding is sized to this

    def score(self, windows_frames: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(text=self.prompts, videos=[list(f) for f in windows_frames],
                                return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            probs = self.model(**inputs).logits_per_video.float().softmax(dim=1)
        return probs[:, :len(self.labels)].cpu().numpy().astype(np.float32)
