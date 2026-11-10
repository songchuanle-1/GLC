"""Run a tiny read-compose-write pass on CPU without external assets."""
import torch

from rpm import DepthWiseWordMemory, RecurrentPhraseMemory


def main():
    torch.random.default_generator.manual_seed(42)
    batch, views, tokens_per_view, words, dim = 2, 2, 4, 3, 32
    visual = torch.randn(batch * views, tokens_per_view, dim)
    text = torch.randn(batch, words, dim)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    read_write = DepthWiseWordMemory(dim, num_heads=4)
    compose_write = RecurrentPhraseMemory(dim, num_heads=4)
    memory = None
    for _ in range(3):
        visual, memory = read_write(
            visual, text, mask, batch, views, tokens_per_view, memory)
        visual, memory = compose_write(
            visual, memory, mask, batch, views, tokens_per_view)
    assert torch.isfinite(visual).all() and torch.isfinite(memory).all()
    assert torch.count_nonzero(memory[1, 2]) == 0
    print("visual shape:", tuple(visual.shape))
    print("memory shape:", tuple(memory.shape))
    print("Finite CPU outputs; padded token state stays zero.")


if __name__ == "__main__":
    main()
