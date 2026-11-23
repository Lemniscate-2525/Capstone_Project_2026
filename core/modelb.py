import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTBlock(nn.Module):

    def __init__(self, dim, n_heads):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        h = self.norm1(x)

        attn_out, _ = self.attention(
            h,
            h,
            h,
            need_weights=False
        )

        x = x + attn_out

        h = self.norm2(x)

        x = x + self.ffn(h)

        return x


class BERTModel(nn.Module):

    def __init__(
        self,
        vocab_size_in,
        d_model_in,
        num_layers_in,
        num_heads_in,
        seq_len_in
    ):

        super().__init__()

        self.seq_len = seq_len_in

        self.tok_emb = nn.Embedding(
            vocab_size_in,
            d_model_in
        )

        self.pos_emb = nn.Embedding(
            seq_len_in,
            d_model_in
        )

        self.layers = nn.ModuleList(
            [
                BERTBlock(
                    d_model_in,
                    num_heads_in
                )
                for _ in range(num_layers_in)
            ]
        )

        self.norm = nn.LayerNorm(
            d_model_in
        )

        self.out = nn.Linear(
            d_model_in,
            1
        )

    def forward(self, x):

        B, T = x.shape

        if T > self.seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds configured maximum {self.seq_len}"
            )

        positions = torch.arange(
            T,
            device=x.device
        )

        h = (
            self.tok_emb(x)
            +
            self.pos_emb(positions).unsqueeze(0)
        )

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        pooled = h.mean(dim=1)

        return self.out(
            pooled
        ).squeeze(-1)