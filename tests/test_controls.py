"""Focused CPU checks for the two prepared innovation interventions.

No full MVGGT construction, checkpoint/data loading, CUDA query or training.
Run in a project environment that already supplies Torch and model imports.
"""

import copy
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from torch import nn
except ImportError as error:
    raise SystemExit('Torch is unavailable; CPU runtime checks were not run.') from error

from rpm.interventions import (
    install_no_composed_write,
    install_no_token_write,
    install_no_visual_read,
    install_self_only_composition,
    install_shared_visual_read,
    self_only_pair_hidden,
    shared_visual_evidence,
)
from rpm.modules import (
    DepthWiseWordMemory,
    RecurrentPhraseMemory,
)
from rpm.fusion import (
    install_plain_fusion,
    parameter_budget,
)


class TinyMemoryPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_word_memory = DepthWiseWordMemory(8, num_heads=2)
        self.recurrent_phrase_memory = RecurrentPhraseMemory(8, num_heads=2)


def inventory(model):
    return [(name, id(value), tuple(value.shape), value.requires_grad)
            for name, value in model.named_parameters()]


class InnovationControlTests(unittest.TestCase):
    def setUp(self):
        # Seed CPU only; these tests never inspect or initialize a CUDA device.
        torch.random.default_generator.manual_seed(41)

    def test_visual_mean_excludes_padding_and_preserves_dtype(self):
        reads = torch.tensor([[[1., 3.], [5., 7.], [float('nan'), 999.]],
                              [[9., 2.], [999., 999.], [999., 999.]]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        actual = shared_visual_evidence(reads, mask)
        expected = torch.tensor([[[3., 5.]] * 3, [[9., 2.]] * 3])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, reads.dtype)
        self.assertTrue(actual.is_contiguous())

    def test_visual_mean_padding_receives_no_gradient(self):
        reads = torch.randn(1, 3, 2, requires_grad=True)
        mask = torch.tensor([[1, 0, 1]])
        shared_visual_evidence(reads, mask).sum().backward()
        expected = torch.tensor([[[1.5, 1.5], [0., 0.], [1.5, 1.5]]])
        torch.testing.assert_close(reads.grad, expected)
        torch.testing.assert_close(
            shared_visual_evidence(reads.detach(), torch.zeros_like(mask)),
            torch.zeros_like(reads))

    def test_self_only_keeps_diagonal_and_removes_source_dependence(self):
        pairs = torch.randn(2, 3, 3, 8, requires_grad=True)
        actual = self_only_pair_hidden(pairs)
        self.assertEqual(actual.shape, pairs.shape)
        self.assertTrue(actual.is_contiguous())
        for receiver in range(3):
            for source in range(3):
                torch.testing.assert_close(actual[:, receiver, source],
                                           pairs[:, receiver, receiver])
        actual.sum().backward()
        expected = torch.zeros_like(pairs)
        for receiver in range(3):
            expected[:, receiver, receiver] = 3
        torch.testing.assert_close(pairs.grad, expected)

    def test_installers_keep_parameter_objects_state_and_cpu_rng(self):
        for installer in (install_shared_visual_read, install_self_only_composition,
                          install_no_visual_read, install_no_token_write,
                          install_no_composed_write):
            model = TinyMemoryPair()
            before_inventory = inventory(model)
            before_state = {name: value.clone() for name, value in model.state_dict().items()}
            before_rng = torch.get_rng_state().clone()
            installer(model)
            self.assertEqual(before_inventory, inventory(model))
            self.assertEqual(list(before_state), list(model.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            with self.assertRaises(ValueError):
                installer(model)

    def test_shared_read_preserves_distinct_prior_states_and_two_writes(self):
        model = TinyMemoryPair()
        install_shared_visual_read(model)
        memory = model.depth_word_memory
        captures = {}
        writes = []
        handles = [
            memory.memory_cell.register_forward_pre_hook(
                lambda _module, args: captures.update(evidence=args[0].detach().clone(),
                                                      hidden=args[1].detach().clone())),
            memory.write_attn.register_forward_hook(lambda *_: writes.append('token')),
            model.recurrent_phrase_memory.write_attn.register_forward_hook(
                lambda *_: writes.append('composed')),
        ]
        x = torch.randn(4, 3, 8)
        text = torch.randn(2, 4, 8)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        prior = torch.randn_like(text).masked_fill(~valid.bool().unsqueeze(-1), 0)
        visual, updated = memory(x, text, valid, 2, 2, 3, prior)
        final_visual, final_memory = model.recurrent_phrase_memory(
            visual, updated, valid, 2, 2, 3)
        evidence = captures['evidence'].reshape_as(text)
        torch.testing.assert_close(evidence[:, 0], evidence[:, 1])
        torch.testing.assert_close(captures['hidden'].reshape_as(prior), prior)
        self.assertFalse(torch.equal(prior[:, 0], prior[:, 1]))
        self.assertEqual(writes, ['token', 'composed'])
        self.assertEqual(final_visual.shape, x.shape)
        self.assertTrue(torch.isfinite(final_memory).all())
        torch.testing.assert_close(updated[~valid.bool()], torch.zeros_like(updated[~valid.bool()]))
        for handle in handles:
            handle.remove()

    def test_self_only_keeps_pair_gradients_and_original_class(self):
        model = TinyMemoryPair()
        phrase = model.recurrent_phrase_memory
        original_forward = RecurrentPhraseMemory.forward
        with torch.no_grad():
            phrase.memory_update.weight.normal_(std=0.2)
            phrase.out_proj.weight.normal_(std=0.2)
        install_self_only_composition(model)
        x = torch.randn(4, 3, 8)
        memory = torch.randn(2, 4, 8, requires_grad=True)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        visual, composed = phrase(x, memory, valid, 2, 2, 3)
        (visual.square().sum() + composed.square().sum()).backward()
        for module in (phrase.pair_dest, phrase.pair_source, phrase.pair_out):
            self.assertIsNotNone(module.weight.grad)
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        self.assertIs(original_forward, RecurrentPhraseMemory.forward)
        self.assertTrue(torch.isfinite(composed).all())
        # Original singleton zero-neighbor behavior is retained.
        single = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        _, singleton_composed = phrase(x, memory.detach(), single, 2, 2, 3)
        torch.testing.assert_close(singleton_composed[:, 0], memory.detach()[:, 0])

    def test_no_visual_read_keeps_token_memory_without_visual_dependency(self):
        model = TinyMemoryPair()
        install_no_visual_read(model)
        word = model.depth_word_memory
        text = torch.randn(2, 4, 8)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        x = torch.randn(4, 3, 8)
        visual, memory = word(x, text, valid, 2, 2, 3)
        _, other_memory = word(x + torch.randn_like(x) * 3, text, valid, 2, 2, 3)
        torch.testing.assert_close(memory, other_memory, rtol=0, atol=0)
        (visual.square().sum() + memory.square().sum()).backward()
        self.assertIsNotNone(word.language_read.in_proj_weight.grad)
        torch.testing.assert_close(word.language_read.in_proj_weight.grad,
                                   torch.zeros_like(word.language_read.in_proj_weight))
        self.assertGreater(word.memory_cell.weight_hh.grad.abs().sum().item(), 0)

    def test_no_token_write_keeps_memory_and_executes_original_branch(self):
        reference = TinyMemoryPair()
        with torch.no_grad():
            reference.depth_word_memory.out_proj.weight.normal_(std=0.2)
        model = copy.deepcopy(reference)
        install_no_token_write(model)
        x = torch.randn(4, 3, 8)
        text = torch.randn(2, 4, 8)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        calls = []
        handle = model.depth_word_memory.write_attn.register_forward_hook(
            lambda *_: calls.append('computed'))
        _, expected_memory = reference.depth_word_memory(x, text, valid, 2, 2, 3)
        visual, memory = model.depth_word_memory(x, text, valid, 2, 2, 3)
        torch.testing.assert_close(visual, x, rtol=0, atol=0)
        torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
        (visual.square().sum() + memory.square().sum()).backward()
        torch.testing.assert_close(model.depth_word_memory.out_proj.weight.grad,
                                   torch.zeros_like(model.depth_word_memory.out_proj.weight))
        self.assertEqual(calls, ['computed'])
        handle.remove()

    def test_no_composed_write_keeps_composed_carry(self):
        reference = TinyMemoryPair()
        with torch.no_grad():
            reference.recurrent_phrase_memory.memory_update.weight.normal_(std=0.2)
            reference.recurrent_phrase_memory.out_proj.weight.normal_(std=0.2)
        model = copy.deepcopy(reference)
        install_no_composed_write(model)
        x = torch.randn(4, 3, 8)
        memory = torch.randn(2, 4, 8)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        _, expected_carry = reference.recurrent_phrase_memory(x, memory, valid, 2, 2, 3)
        visual, carry = model.recurrent_phrase_memory(x, memory, valid, 2, 2, 3)
        torch.testing.assert_close(visual, x, rtol=0, atol=0)
        torch.testing.assert_close(carry, expected_carry, rtol=0, atol=0)
        (visual.square().sum() + carry.square().sum()).backward()
        torch.testing.assert_close(model.recurrent_phrase_memory.out_proj.weight.grad,
                                   torch.zeros_like(model.recurrent_phrase_memory.out_proj.weight))
        self.assertGreater(model.recurrent_phrase_memory.pair_out.weight.grad.abs().sum().item(), 0)

    def test_plain_fusion_fixed_parameter_budget_and_cpu_rng(self):
        model = TinyMemoryPair()
        before_rng = torch.get_rng_state().clone()
        expected = parameter_budget(8)
        original = sum(p.numel() for p in model.parameters())
        self.assertEqual(original, expected['rpm_module_parameters'])
        install_plain_fusion(model)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         expected['plain_module_parameters'])
        self.assertEqual(expected['difference'], -4)
        self.assertFalse(any('.encoder.' in name for name, _ in model.named_parameters()))

    def test_plain_fusion_shape_gradients_and_no_memory_carry(self):
        model = TinyMemoryPair()
        install_plain_fusion(model)
        # Make visual writes nonzero so all ordinary branch graph paths are
        # observed; initialization itself still uses zero visual projections.
        with torch.no_grad():
            model.depth_word_memory.out_proj.weight.normal_(std=0.2)
            model.recurrent_phrase_memory.out_proj.weight.normal_(std=0.2)
        x = torch.randn(4, 3, 8)
        text = torch.randn(2, 4, 8)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        first, state = model.depth_word_memory(x, text, valid, 2, 2, 3,
                                               torch.randn_like(text))
        _, state_with_other_prior = model.depth_word_memory(x, text, valid, 2, 2, 3,
                                                           torch.randn_like(text))
        torch.testing.assert_close(state, state_with_other_prior, rtol=0, atol=0)
        visual, final_state = model.recurrent_phrase_memory(first, state, valid, 2, 2, 3)
        self.assertEqual(visual.shape, x.shape)
        self.assertTrue(torch.isfinite(final_state).all())
        (visual.square().sum() + final_state.square().sum()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
