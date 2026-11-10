"""Local, parameter-preserving interventions for the original M93 modules.

These functions create no parameters, consume no random draws and load no assets.
They are implementation preparation, not a scientific result or launch route.
"""

from types import MethodType

import torch


def shared_visual_evidence(visual_context, attention_mask):
    """Broadcast the valid-token visual-read mean without pooling token states.

    Valid special/subword tokens follow the original tokenizer attention mask.
    Padded read values are excluded before summing.  A wholly padded row yields
    zero here; the original complete module still assumes at least one valid
    token, so this helper does not add a new all-padding model-input guarantee.
    """
    if visual_context.ndim != 3 or attention_mask.shape != visual_context.shape[:2]:
        raise ValueError('Expected visual_context[B,T,D] and attention_mask[B,T]')
    valid = attention_mask.bool().unsqueeze(-1)
    total = visual_context.masked_fill(~valid, 0).sum(dim=1, keepdim=True)
    denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)
    shared = total / denominator
    return shared.expand_as(visual_context).contiguous()


def shared_visual_read_forward(self, x, text_embeds, attention_mask, batch_size,
                               num_views, tokens_per_view, memory=None):
    """Original DepthWiseWordMemory flow with only visual_context broadcast.

    The mask is a local forward argument rather than cached hook state, so
    concurrent calls and activation-checkpoint recomputation do not share it.
    """
    tokens = x.reshape(batch_size, num_views * tokens_per_view, -1)
    normalized_tokens = self.token_norm(tokens)
    valid = attention_mask.bool().unsqueeze(-1)
    if memory is None:
        memory = self.language_norm(text_embeds).masked_fill(~valid, 0)
    visual_context, _ = self.language_read(
        self.memory_norm(memory), normalized_tokens, normalized_tokens,
        need_weights=False,
    )
    visual_context = shared_visual_evidence(visual_context, attention_mask)
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


def install_shared_visual_read(model):
    """Change only one model instance, retaining existing modules and weights."""
    word_memory = model.depth_word_memory
    if getattr(word_memory, '_innovation_shared_visual_read', False):
        raise ValueError('Shared-visual-read intervention is already installed')
    # Binding a forward method adds neither a module nor a parameter.  All
    # state_dict/optimizer names and post-constructor RNG state remain intact.
    word_memory.forward = MethodType(shared_visual_read_forward, word_memory)
    word_memory._innovation_shared_visual_read = True


def self_only_pair_hidden(pair_hidden):
    """Replace Wdest*g_i + Wsrc*g_j by Wdest*g_i + Wsrc*g_i.

    Keep the original [B,T,T,D] workload and contiguous layout.  Subsequent
    off-diagonal/padding masks and zero-neighbor handling are not changed.
    """
    if pair_hidden.ndim != 4 or pair_hidden.shape[1] != pair_hidden.shape[2]:
        raise ValueError('Expected square pair_hidden[B,T,T,D]')
    receiver_only = pair_hidden.diagonal(dim1=1, dim2=2).movedim(-1, 1)
    return receiver_only.unsqueeze(2).expand_as(pair_hidden).contiguous()


def _self_only_activation_input(_module, args):
    if len(args) != 1:
        raise ValueError('Original pair activation must receive one tensor')
    return (self_only_pair_hidden(args[0]),)


def install_self_only_composition(model):
    """Preserve both writes and all pair projections; remove explicit exchange.

    Destination/source projections become functionally redundant for this
    condition.  Parameter-count equality does not imply equal function classes,
    and contextual text embeddings still carry cross-token information.
    """
    phrase_memory = model.recurrent_phrase_memory
    if hasattr(phrase_memory, '_innovation_self_only_handle'):
        raise ValueError('Self-only-composition intervention is already installed')
    phrase_memory._innovation_self_only_handle = (
        phrase_memory.pair_activation.register_forward_pre_hook(
            _self_only_activation_input))


def _zero_visual_context(_module, _args, output):
    context, attention_weights = output
    return context * 0.0, attention_weights


def install_no_visual_read(model):
    """Compute the added visual read then gate its context before GRU.

    The original PWAM/image paths remain.  This is not a globally nonvisual
    model.  Read-branch parameters remain registered but their gradients through
    this output are zero; do not call their stored count active capacity.
    """
    word_memory = model.depth_word_memory
    if hasattr(word_memory, '_innovation_no_visual_read_handle'):
        raise ValueError('No-visual-read intervention is already installed')
    word_memory._innovation_no_visual_read_handle = (
        word_memory.language_read.register_forward_hook(_zero_visual_context))


def _zero_residual(_module, _args, output):
    # Multiplication retains the executed graph with zero branch gradients.
    # It does not sanitize a NaN/Inf or pretend inactive weights are optimized
    # functional capacity.
    return output * 0.0


def install_no_token_write(model):
    """Gate only token visual residual; preserve read, state and composition."""
    word_memory = model.depth_word_memory
    if hasattr(word_memory, '_innovation_no_token_write_handle'):
        raise ValueError('No-token-write intervention is already installed')
    word_memory._innovation_no_token_write_handle = (
        word_memory.out_proj.register_forward_hook(_zero_residual))


def install_no_composed_write(model):
    """Gate only composed visual residual; keep composed state for next depth."""
    phrase_memory = model.recurrent_phrase_memory
    if hasattr(phrase_memory, '_innovation_no_composed_write_handle'):
        raise ValueError('No-composed-write intervention is already installed')
    phrase_memory._innovation_no_composed_write_handle = (
        phrase_memory.out_proj.register_forward_hook(_zero_residual))
