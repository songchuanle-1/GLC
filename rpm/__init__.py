"""GLC memory modules in the historical, checkpoint-compatible rpm namespace."""
from .modules import DepthWiseWordMemory, RecurrentPhraseMemory

__version__ = "0.1.0"
__all__ = ["DepthWiseWordMemory", "RecurrentPhraseMemory"]
