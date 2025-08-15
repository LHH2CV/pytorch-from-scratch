import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        # 1. Causal mask (prevent looking ahead)
        # Mask shape: [T, T] with True for "illegal" positions
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # 2. Masked self-attention
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)

        # 3. Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class SimpleGPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, input_ids):
        """
        input_ids: [B, T]  - token IDs
        """
        B, T = input_ids.shape
        assert T <= self.max_len

        # 1. Token embedding + Position embedding
        tok_emb = self.token_embed(input_ids)                  # [B, T, D]
        pos_ids = torch.arange(T, device=input_ids.device)    # [T]
        pos_emb = self.pos_embed(pos_ids)[None, :, :]          # [1, T, D]
        x = tok_emb + pos_emb                                  # [B, T, D]

        # 2. Masked Decoder Layers
        for layer in self.layers:
            x = layer(x)

        # 3. Final projection to logits
        x = self.ln_final(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits
