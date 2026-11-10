"""Stdlib-only integration contracts; no upstream source is stored in fixtures.

Optional local equivalence audit:
set RPM_TEST_UPSTREAM_SOURCE and RPM_TEST_REFERENCE_MODEL to existing files.
These environment variables are read only; generated code stays in memory.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("integrate_mvggt", ROOT / "tools/integrate_mvggt.py")
integration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = integration
SPEC.loader.exec_module(integration)


def fixture(*, guarded=True):
    """An original, non-runnable interface sketch, not an upstream implementation."""
    initializer = """
        pi3_path = os.path.join(pretrained_model_name_or_path, 'synthetic-asset')
        self.multimodal_decoder.load_state_dict({})
"""
    if guarded:
        initializer = """
        if pretrained_model_name_or_path is None:
            pass
        else:
            pi3_path = os.path.join(pretrained_model_name_or_path, 'synthetic-asset')
            self.multimodal_decoder.load_state_dict({})
"""
    # Only public interface names and minimal mathematical layout anchors are
    # represented. Placeholder calls intentionally make this non-runnable.
    source = '''"""Handwritten synthetic interface."""
from __future__ import annotations
from .synthetic import Placeholder
from ..utils.synthetic import AnotherPlaceholder

class MVGGT(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, use_referring_segmentation=True):
        self.synthetic_before = construct_marker()
        if self.use_referring_segmentation:
            dec_embed_dim = 8
            num_injections = 2
            self.controlnet_injectors = nn.ModuleList()
            for counter in range(num_injections):
                self.controlnet_injectors.append(make_injector(dec_embed_dim))
INITIALIZER
        self.synthetic_after = finalize_marker()

    def decode(self, hidden, N, H, W, text_embeds=None, attention_mask=None):
        BN, hw, unused = hidden.shape
        B = BN // N
        hidden = add_synthetic_registers(hidden)
        hw = hidden.shape[1]
        for i in range(len(self.decoder)):
            if synthetic_local_condition(i):
                multimodal_hidden = multimodal_hidden.reshape(B * N, hw, -1)
            else:
                multimodal_hidden = multimodal_hidden.reshape(B, N * hw, -1)
            preserve_checkpoint_dispatch(i)
            if self.use_referring_segmentation:
                if i in self.layer_indices_map:
                    x_residual = fusion_module(multimodal_hidden, text_embeds_, attention_mask_)
                    multimodal_hidden = multimodal_hidden + self.res_gate(x_residual) * x_residual
                    multimodal_hidden_reshaped = multimodal_hidden.reshape(B * N, hw, -1)
                    retain_mask_and_injection_order(multimodal_hidden_reshaped)
        return synthetic_result()
'''
    return source.replace("INITIALIZER\n", textwrap.indent(initializer.lstrip("\n"), "    "))


def model(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


class IntegrationTests(unittest.TestCase):
    def test_exact_insertion_order_and_no_other_statement_changes(self):
        original = ast.parse(fixture())
        result = integration.integrate_source(fixture())
        self.assertEqual(result.pi3_initialization, "guarded")
        generated = ast.parse(result.source)
        cls = model(generated, "MVGGT_M93_RPM")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        branch = next(n for n in init.body if isinstance(n, ast.If))
        offset = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For))
        self.assertEqual([integration._path(n.targets[0]) for n in branch.body[offset + 1:offset + 3]],
                         ["self.depth_word_memory", "self.recurrent_phrase_memory"])
        self.assertIsInstance(branch.body[offset + 3], ast.If)
        decode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
        loop_index = next(i for i, n in enumerate(decode.body) if isinstance(n, ast.For))
        state = decode.body[loop_index - 1]
        self.assertEqual(ast.unparse(state), "word_memory = None")
        fusion = next(n for n in ast.walk(decode) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Compare))
        calls = fusion.body[2:4]
        self.assertEqual([integration._path(n.value.func) for n in calls],
                         ["self.depth_word_memory", "self.recurrent_phrase_memory"])
        self.assertEqual([integration._path(a) for a in calls[0].value.args],
                         ["multimodal_hidden", "text_embeds", "attention_mask", "B", "N", "hw", "word_memory"])
        self.assertEqual([integration._path(a) for a in calls[1].value.args],
                         ["multimodal_hidden", "word_memory", "attention_mask", "B", "N", "hw"])
        # Remove only our additions. Every remaining AST node must match the
        # supplied upstream module after its location-independent imports.
        del fusion.body[2:4]
        del decode.body[loop_index - 1]
        del branch.body[offset + 1:offset + 3]
        cls.name = "MVGGT"
        generated.body = [n for n in generated.body
                          if not (isinstance(n, ast.ImportFrom) and n.module == "rpm.modules")]
        integration.resolve_relative_imports(original)
        self.assertEqual(dump(generated), dump(original))
        compile(result.source, "<generated>", "exec")

    def test_relative_imports_and_future_statement_remain_legal(self):
        tree = ast.parse(integration.integrate_source(fixture()).source)
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
        self.assertEqual([n.module for n in imports],
                         ["__future__", "rpm.modules", "mvggt.models.synthetic", "mvggt.utils.synthetic"])
        self.assertTrue(all(n.level == 0 for n in imports))

    def test_user_supplied_leading_notices_are_preserved(self):
        preamble = "# Synthetic author notice for this test.\n# Synthetic usage terms.\n\n"
        for source in (fixture(), fixture().split("\n", 1)[1]):
            result = integration.integrate_source(preamble + source)
            self.assertTrue(result.source.startswith(preamble))
            self.assertEqual(dump(ast.parse(result.source)),
                             dump(ast.parse(integration.integrate_source(source).source)))

    def test_unguarded_pi3_initializer_is_preserved_and_reported(self):
        result = integration.integrate_source(fixture(guarded=False))
        self.assertEqual(result.pi3_initialization, "requires_path")
        self.assertNotIn("pretrained_model_name_or_path is None", result.source)

    def test_unsupported_or_ambiguous_anchors_fail_closed(self):
        mutations = [
            ("class MVGGT(nn.Module):", "class OtherModel(nn.Module):"),
            ("self.controlnet_injectors = nn.ModuleList()", "self.controlnet_injectors = nn.ModuleList([existing])"),
            ("range(num_injections)", "range(other_count)"),
            ("B = BN // N", "B = BN / N"),
            ("B, N * hw, -1", "B, hw, -1"),
            ("attention_mask_)", "other_mask)"),
            ("self.res_gate(x_residual) * x_residual", "self.res_gate(x_residual) + x_residual"),
            ("self.multimodal_decoder.load_state_dict({})", "some_other_load({})"),
            ("pretrained_model_name_or_path is None", "pretrained_model_name_or_path is not None"),
            ("preserve_checkpoint_dispatch(i)", "word_memory = stale_state"),
            ("def decode(self, hidden, N, H, W,", "def decode(self, hidden, H, N, W,"),
            ("from .synthetic import", "from ...synthetic import"),
        ]
        for old, new in mutations:
            with self.subTest(anchor=old):
                self.assertIn(old, fixture())
                with self.assertRaises(integration.IntegrationError):
                    integration.integrate_source(fixture().replace(old, new))
        with self.assertRaises(integration.IntegrationError):
            integration.integrate_source(fixture() + "\nclass MVGGT(nn.Module):\n    pass\n")
        with self.assertRaises(integration.IntegrationError):
            integration.integrate_source("def invalid(")
        with self.assertRaises(integration.IntegrationError):
            integration.integrate_source(fixture() + "\nfrom other import Existing as RecurrentPhraseMemory\n")

    def test_files_require_explicit_output_and_preserve_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.py", root / "generated/model.py"
            source.write_text(fixture(), encoding="utf-8")
            original = source.read_bytes()
            integration.integrate_file(source, output)
            generated = output.read_bytes()
            with self.assertRaises(FileExistsError):
                integration.integrate_file(source, output)
            self.assertEqual(output.read_bytes(), generated)
            output.write_text("old artifact", encoding="utf-8")
            integration.integrate_file(source, output, overwrite=True)
            self.assertEqual(output.read_bytes(), generated)
            with self.assertRaises(integration.IntegrationError):
                integration.integrate_file(source, source, overwrite=True)
            self.assertEqual(source.read_bytes(), original)

    def test_invalid_source_never_creates_or_overwrites_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.py", root / "output.py"
            source.write_text("class Unsupported: pass", encoding="utf-8")
            with self.assertRaises(integration.IntegrationError):
                integration.integrate_file(source, output)
            self.assertFalse(output.exists())
            output.write_text("preserve me", encoding="utf-8")
            with self.assertRaises(integration.IntegrationError):
                integration.integrate_file(source, output, overwrite=True)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve me")


@unittest.skipUnless(os.environ.get("RPM_TEST_UPSTREAM_SOURCE")
                     and os.environ.get("RPM_TEST_REFERENCE_MODEL"),
                     "Local upstream/reference paths were not supplied")
class LocalReferenceEquivalenceTests(unittest.TestCase):
    def test_generated_model_matches_local_original_m93_ast(self):
        upstream = Path(os.environ["RPM_TEST_UPSTREAM_SOURCE"])
        reference = Path(os.environ["RPM_TEST_REFERENCE_MODEL"])
        generated = ast.parse(integration.integrate_source(upstream.read_text(encoding="utf-8-sig")).source)
        expected = ast.parse(reference.read_text(encoding="utf-8-sig"))
        self.assertEqual(dump(model(generated, "MVGGT_M93_RPM")),
                         dump(model(expected, "MVGGT_M93_RPM")))
        # The original keeps the two extension classes in the model file;
        # generation imports them from this repository instead.
        expected.body = [n for n in expected.body if not
                         (isinstance(n, ast.ClassDef) and n.name in
                          {"DepthWiseWordMemory", "RecurrentPhraseMemory"})]
        generated.body = [n for n in generated.body if not
                          (isinstance(n, ast.ImportFrom) and n.module == "rpm.modules")]
        integration.resolve_relative_imports(expected)
        self.assertEqual(dump(generated), dump(expected))


if __name__ == "__main__":
    unittest.main()
