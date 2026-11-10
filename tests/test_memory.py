"""Small CPU-only tests of the original phrase memory module."""
import unittest
import torch
from rpm.modules import RecurrentPhraseMemory as Memory

class RecurrentPhraseMemoryTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(2323)
        torch.set_num_threads(2)
        self.module = Memory(32)
        self.x = torch.randn(2, 8, 32)
        self.memory = torch.randn(2, 2, 32)
        self.mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    def call(self, x=None, memory=None, mask=None):
        return self.module(self.x if x is None else x, self.memory if memory is None else memory, self.mask if mask is None else mask, 2, 2, 4)

    def activate(self):
        with torch.no_grad():
            for module in (self.module.pair_dest, self.module.pair_source, self.module.pair_out, self.module.memory_update, self.module.out_proj):
                module.weight.normal_(0, 0.05)
                if module.bias is not None:
                    module.bias.normal_(0, 0.02)

    def test_identity_and_memory_preservation(self):
        output, memory = self.call()
        self.assertTrue(torch.equal(output, self.x))
        self.assertTrue(torch.equal(memory[:, 0], self.memory[:, 0]))
        self.assertTrue(torch.equal(memory[1, 1], torch.zeros_like(memory[1, 1])))

    def test_source_word_changes_destination_and_residual(self):
        self.activate()
        reference, reference_memory = self.call()
        changed_memory = self.memory.clone()
        changed_memory[:, 1] += 2.0
        output, memory = self.call(memory=changed_memory)
        self.assertGreater((memory[:, 0] - reference_memory[:, 0]).abs().sum().item(), 0)
        self.assertFalse(torch.allclose(output, reference))

    def test_padding_word_is_ignored(self):
        self.activate()
        reference, reference_memory = self.call()
        changed_memory = self.memory.clone()
        changed_memory[1, 1] = 100.0
        output, memory = self.call(memory=changed_memory)
        self.assertTrue(torch.allclose(output[1], reference[1], atol=1e-06))
        self.assertTrue(torch.allclose(memory[1], reference_memory[1], atol=1e-06))

    def test_permutation_equivariance(self):
        self.activate()
        output, memory = self.call()
        permuted, permuted_memory = self.call(memory=self.memory[:, [1, 0]], mask=self.mask[:, [1, 0]])
        self.assertTrue(torch.allclose(output, permuted, atol=1e-06))
        self.assertTrue(torch.allclose(memory[:, [1, 0]], permuted_memory, atol=1e-06))

    def test_single_word_has_no_pair_update(self):
        self.activate()
        memory = self.memory[:, :1]
        _, output_memory = self.call(memory=memory, mask=torch.ones(2, 1, dtype=torch.bool))
        self.assertTrue(torch.allclose(output_memory, memory))

    def test_twelve_depth_backward(self):
        self.activate()
        initial_x = self.x.clone().requires_grad_(True)
        initial_memory = self.memory.clone().requires_grad_(True)
        x = initial_x
        memory = initial_memory
        for _ in range(12):
            x, memory = self.call(x=x, memory=memory)
        (x.square().mean() + memory.square().mean()).backward()
        self.assertIsNotNone(initial_x.grad)
        self.assertIsNotNone(initial_memory.grad)
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
