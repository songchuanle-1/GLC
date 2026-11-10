"""Original RPM research extension; external backbones are not bundled."""

import torch
from torch import nn

class PlainReadWriteFusion(nn.Module):
    """Ordinary text-to-visual read, token FFN and visual residual write.

    Reconstruct token context from text at each depth; the incoming memory
    argument is intentionally unused. There is no GRU or explicit pair MLP.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        hidden = 2 * dim
        self.token_norm = nn.LayerNorm(dim)
        self.language_norm = nn.LayerNorm(dim)
        self.language_read = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.evidence_norm = nn.LayerNorm(dim)
        self.token_ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                       nn.Linear(hidden, dim))
        self.updated_language_norm = nn.LayerNorm(dim)
        self.write_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, text_embeds, attention_mask, batch_size, num_views,
                tokens_per_view, memory=None):
        tokens = x.reshape(batch_size, num_views * tokens_per_view, -1)
        valid = attention_mask.bool().unsqueeze(-1)
        normalized_tokens = self.token_norm(tokens)
        initial = self.language_norm(text_embeds).masked_fill(~valid, 0)
        evidence, _ = self.language_read(initial, normalized_tokens,
                                         normalized_tokens, need_weights=False)
        state = (initial + self.token_ffn(self.evidence_norm(evidence)))
        state = state.masked_fill(~valid, 0)
        language = self.updated_language_norm(text_embeds + state)
        update, _ = self.write_attn(normalized_tokens, language, language,
                                    key_padding_mask=~attention_mask.bool(),
                                    need_weights=False)
        return (tokens + self.out_proj(update)).reshape_as(x), state

class PlainTokenAttentionWrite(nn.Module):
    """Standard token self-attention/FFN followed by a visual residual write."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        hidden = dim + dim // 2
        self.memory_norm = nn.LayerNorm(dim)
        self.token_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.token_ffn_norm = nn.LayerNorm(dim)
        self.token_ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                       nn.Linear(hidden, dim))
        self.token_norm = nn.LayerNorm(dim)
        self.write_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, memory, attention_mask, batch_size, num_views,
                tokens_per_view):
        tokens = x.reshape(batch_size, num_views * tokens_per_view, -1)
        valid = attention_mask.bool()
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(dim=1)
        normalized = self.memory_norm(memory).masked_fill(~valid.unsqueeze(-1), 0)
        context, _ = self.token_attn(normalized, normalized, normalized,
                                     key_padding_mask=~safe_valid,
                                     need_weights=False)
        state = memory + context
        state = state + self.token_ffn(self.token_ffn_norm(state))
        state = state.masked_fill(~valid.unsqueeze(-1), 0)
        update, _ = self.write_attn(self.token_norm(tokens), state, state,
                                    key_padding_mask=~safe_valid,
                                    need_weights=False)
        residual = self.out_proj(update) * valid.any(dim=1)[:, None, None]
        return (tokens + residual).reshape_as(x), state

def parameter_budget(dim):
    """Declared inventory, not a measured FLOP or training-optimizer claim."""
    first_hidden = 2 * dim
    second_hidden = dim + dim // 2
    rpm = 25 * dim * dim + 38 * dim
    plain = (18 * dim * dim + 2 * dim * (first_hidden + second_hidden)
             + first_hidden + second_hidden + 34 * dim)
    return {'dimension': dim, 'rpm_module_parameters': rpm,
            'plain_module_parameters': plain, 'difference': plain - rpm,
            'relative_difference': (plain - rpm) / rpm,
            'first_ffn_width': first_hidden, 'second_ffn_width': second_hidden,
            'compute_matched': False,
            'scope': 'Fixed ordinary fusion widths; actual operation counts must be reported.'}

def install_plain_fusion(model, initialization_seed=1111):
    """Replace only added memory modules after original common initialization.

    CPU-only fork restores the original post-constructor CPU RNG state. Explicit
    devices=[] prevents fork_rng from enumerating CUDA devices. The original
    common model parameters are not reset, renamed or replaced.
    """
    reference = model.depth_word_memory.out_proj.weight
    dim = reference.shape[0]
    heads = model.depth_word_memory.language_read.num_heads
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initialization_seed)
        word = PlainReadWriteFusion(dim, heads)
        token = PlainTokenAttentionWrite(dim, heads)
    word.to(device=reference.device, dtype=reference.dtype)
    token.to(device=reference.device, dtype=reference.dtype)
    word.train(model.depth_word_memory.training)
    token.train(model.recurrent_phrase_memory.training)
    specification = parameter_budget(dim)
    actual = sum(p.numel() for module in (word, token) for p in module.parameters())
    if actual != specification['plain_module_parameters']:
        raise ValueError('Ordinary fusion parameter inventory differs from declared fixed widths')
    model.depth_word_memory = word
    model.recurrent_phrase_memory = token
    model._innovation_plain_parameter_spec = specification
