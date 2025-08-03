import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Gemma3TextModel

class GemmaAdapter(nn.Module):
    """
    Frozen Gemma → 768-D text vector that TRELLIS already expects.
    """
    def __init__(
        self,
        model_name: str = "google/gemma-3b",
        max_len: int = 256,
        out_dim: int = 768
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gemma = Gemma3TextModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(torch.float16)  # use float16 for Gemma
        self.gemma.eval().requires_grad_(False)       # keep it frozen

        # --- tiny trainable head ---
        self.proj = nn.Sequential(
            nn.Linear(self.gemma.config.hidden_size, out_dim, bias=False),
            nn.LayerNorm(out_dim)                 # handles Gemma↔CLIP norm mismatch
        )
        self.max_len = max_len

    @torch.no_grad()          # inference mode for backbone
    def _embed(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        ).to(self.gemma.device)
        h = self.gemma(**toks).last_hidden_state       # [B, L, 4096]
        return h[:, 0]                          # CLS pooling

    def forward(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            pooled = self._embed(texts)                # freeze Gemma
        return self.proj(pooled)                     # train only this