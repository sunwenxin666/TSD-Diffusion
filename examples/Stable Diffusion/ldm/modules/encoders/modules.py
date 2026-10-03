import os
import logging
import warnings
import torch
import torch.nn as nn
from einops import repeat
from transformers import CLIPTokenizer, CLIPTextModel

from ldm.modules.x_transformer import Encoder, TransformerWrapper

# Suppress transformers warnings about unused weights
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size 30522"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses TransformerWrapper to embed tokens."""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_length=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_length,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """
    HuggingFace CLIP Text Encoder kept on CPU; output embeddings moved to out_device (GPU).
    Supports local path: version can be a directory containing tokenizer + text model files.
    """
    def __init__(
        self,
        version="examples/Stable Diffusion/models/clip-vit-large-patch14",
        cpu_device="cpu",
        out_device="cuda",
        max_length=77,
        device=None,           # legacy: treat as out_device
        output_dtype=None,     # optional: torch.float16 / torch.bfloat16
        **kwargs
    ):
        super().__init__()

        if device is not None:
            out_device = device

        version = os.getenv("CLIP_MODEL_PATH", version)

        self.cpu_device = torch.device(cpu_device)
        self.out_device = torch.device(out_device)
        self.max_length = max_length
        self.output_dtype = output_dtype

        # version can be local folder or HF repo id
        if isinstance(version, str) and os.path.isdir(version):
            pretrained_path = version
        else:
            pretrained_path = version

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path)
        self.transformer = CLIPTextModel.from_pretrained(
            pretrained_path,
            low_cpu_mem_usage=True
        ).to(self.cpu_device)

        self.freeze()

    def freeze(self):
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, text):
        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # IMPORTANT: tokens on CPU (same as model)
        tokens = batch["input_ids"].to(self.cpu_device)
        
        # Ensure model is on the correct device
        if self.transformer.device != self.cpu_device:
            self.transformer = self.transformer.to(self.cpu_device)

        outputs = self.transformer(input_ids=tokens)
        z_cpu = outputs.last_hidden_state  # CPU

        # Boundary move to GPU
        if self.output_dtype is not None:
            z = z_cpu.to(self.out_device, dtype=self.output_dtype, non_blocking=True)
        else:
            z = z_cpu.to(self.out_device, non_blocking=True)
        return z

    def encode(self, text):
        return self(text)
