"""Original RPM research extension; external backbones are not bundled."""

import torch
from torch import nn

class DepthWiseWordMemory(nn.Module):
    """Carry one visually grounded hidden state per word across fusion depth."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.language_norm = nn.LayerNorm(dim)
        self.language_read = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)
        self.language_update = nn.Linear(dim, dim)
        self.updated_language_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_cell = nn.GRUCell(dim, dim)
        self.write_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.language_update.weight)
        nn.init.zeros_(self.language_update.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, text_embeds, attention_mask, batch_size, num_views,
                tokens_per_view, memory=None):
        tokens = x.reshape(batch_size, num_views * tokens_per_view, -1)
        normalized_tokens = self.token_norm(tokens)
        valid = attention_mask.bool().unsqueeze(-1)
        if memory is None:
            memory = self.language_norm(text_embeds).masked_fill(~valid, 0)
        visual_context, _ = self.language_read(
            self.memory_norm(memory), normalized_tokens, normalized_tokens,
            need_weights=False,
        )
        # Flatten only batch/word axes; each word keeps its own recurrent state.
        memory = self.memory_cell(
            self.evidence_norm(visual_context).reshape(-1, tokens.shape[-1]),
            memory.reshape(-1, tokens.shape[-1]),
        ).reshape_as(text_embeds).masked_fill(~valid, 0)
        updated_language = text_embeds + self.language_update(memory)
        updated_language = self.updated_language_norm(updated_language)
        update, _ = self.write_attn(
            normalized_tokens, updated_language, updated_language,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        return (tokens + self.out_proj(update)).reshape_as(x), memory

class RecurrentPhraseMemory(nn.Module):
    """Compose ordered word-pair relations inside recurrent depth memory."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.pair_dest = nn.Linear(dim, dim)
        self.pair_source = nn.Linear(dim, dim)
        self.pair_out = nn.Linear(dim, dim)
        self.pair_activation = nn.GELU()
        self.memory_update = nn.Linear(dim, dim, bias=False)
        self.write_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.memory_update.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, memory, attention_mask, batch_size, num_views,
                tokens_per_view):
        tokens = x.reshape(batch_size, num_views * tokens_per_view, -1)
        valid = attention_mask.bool()
        valid_mask = valid.unsqueeze(-1)
        grounded = self.memory_norm(memory).masked_fill(~valid_mask, 0)

        pair_hidden = (
            self.pair_dest(grounded).unsqueeze(2)
            + self.pair_source(grounded).unsqueeze(1)
        )
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        pair_valid = pair_valid & ~torch.eye(
            grounded.shape[1], dtype=torch.bool, device=x.device).unsqueeze(0)
        messages = self.pair_out(self.pair_activation(pair_hidden))
        messages = messages.masked_fill(~pair_valid.unsqueeze(-1), 0)
        relations = messages.sum(2) / pair_valid.sum(2).clamp_min(1).unsqueeze(-1)

        composed_memory = memory + self.memory_update(relations)
        composed_memory = composed_memory.masked_fill(~valid_mask, 0)

        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(dim=1)
        update, _ = self.write_attn(
            self.token_norm(tokens), composed_memory, composed_memory,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        residual = self.out_proj(update) * valid.any(dim=1)[:, None, None]
        return (tokens + residual).reshape_as(x), composed_memory
