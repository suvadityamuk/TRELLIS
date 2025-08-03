import torch, torch.nn as nn
from transformers import T5Tokenizer, UMT5EncoderModel

class T5Adapter(nn.Module):
    """
    Frozen T5 encoder → 768-D text vector for TRELLIS.
    """
    def __init__(self, model_name="google/umt5-base"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512
        self.umt5 = UMT5EncoderModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).eval()
        self.umt5.requires_grad_(False)          # freeze
        assert self.umt5.config.d_model == 768,  \
               "Use a 768-dim T5 (base) or add a Linear proj here"
        
        self.queries = nn.Parameter(torch.randn(77, 768))  # 77 tokens
        self.attn = nn.MultiheadAttention(768, 8, batch_first=True)
        self.ln = nn.LayerNorm(768)

    @torch.no_grad()
    def _encode_long(self, texts):
        toks = self.tok(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(self.umt5.device)
        return self.umt5(**toks).last_hidden_state

    def forward(self, texts):
        h = self._encode_long(texts)                    # context
        B = h.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) # (B, 77, 768)
        pooled, _ = self.attn(q, h, h)                  # cross-attend
        return self.ln(pooled)