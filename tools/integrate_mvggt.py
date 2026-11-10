"""Generate a local MVGGT integration from a separately obtained upstream file.

This repository contains no upstream model implementation. The generated file
contains the user's upstream code and inherits its applicable restrictions.
The transformer never imports or executes that code, Torch, or a CUDA API.

Only the supported MVGGT architecture is accepted. Relative imports are resolved
as if the input module were ``mvggt.models.mvggt_training``. Existing Pi3 guards
are preserved; an older unguarded initializer still requires a Pi3 path.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


class IntegrationError(ValueError):
    """The supplied source does not satisfy the supported integration contract."""


@dataclass(frozen=True)
class IntegrationResult:
    source: str
    pi3_initialization: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrationError(message)


def _one(items, label: str):
    items = list(items)
    _require(len(items) == 1, f"Expected exactly one {label}; found {len(items)}")
    return items[0]


def _path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _assignment(node: ast.AST, target: str) -> bool:
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and _path(node.targets[0]) == target)


def _call(node: ast.AST, target: str) -> bool:
    return isinstance(node, ast.Call) and _path(node.func) == target


def _same(node: ast.AST, other: ast.AST) -> bool:
    return ast.dump(node, include_attributes=False) == ast.dump(other, include_attributes=False)


def _name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _method(model: ast.ClassDef, name: str) -> ast.FunctionDef:
    return _one((n for n in model.body if isinstance(n, ast.FunctionDef)
                 and n.name == name), f"{name} method")


def _require_args(function: ast.FunctionDef, expected: list[str]) -> None:
    args = function.args
    _require([a.arg for a in args.posonlyargs + args.args] == expected
             and not args.vararg and not args.kwarg and not args.kwonlyargs,
             f"Unsupported {function.name} argument layout")


def _pi3_mode(statements: list[ast.stmt]) -> str:
    _require(bool(statements), "Missing Pi3 initialization after injectors")
    first = statements[0]
    mode = "requires_path"
    body = statements
    if isinstance(first, ast.If):
        test = first.test
        _require(len(statements) == 1 and isinstance(test, ast.Compare)
                 and _name(test.left, "pretrained_model_name_or_path")
                 and len(test.ops) == len(test.comparators) == 1
                 and isinstance(test.ops[0], ast.Is) and _none(test.comparators[0])
                 and bool(first.orelse), "Unsupported Pi3 initialization guard")
        body = first.orelse
        mode = "guarded"
    _require(bool(body) and _assignment(body[0], "pi3_path")
             and _call(body[0].value, "os.path.join")
             and body[0].value.args
             and _name(body[0].value.args[0], "pretrained_model_name_or_path"),
             "Unsupported Pi3 path anchor")
    _one((n for statement in body for n in ast.walk(statement)
          if _call(n, "self.multimodal_decoder.load_state_dict")),
         "multimodal Pi3 state load")
    return mode


def _constructor_anchor(model: ast.ClassDef):
    init = _method(model, "__init__")
    args = init.args.posonlyargs + init.args.args
    _require(not init.args.vararg and not init.args.kwarg
             and {"self", "pretrained_model_name_or_path", "use_referring_segmentation"}
             <= {a.arg for a in args}, "Unsupported constructor arguments")
    branch = _one((n for n in init.body if isinstance(n, ast.If)
                   and _path(n.test) == "self.use_referring_segmentation"),
                  "referring-segmentation constructor branch")
    _require(not branch.orelse, "Constructor referring branch has an unsupported else")
    setup = _one((n for n in branch.body
                  if _assignment(n, "self.controlnet_injectors")), "injector collection")
    _require(_call(setup.value, "nn.ModuleList") and not setup.value.args
             and not setup.value.keywords, "Expected an initially empty injector collection")
    index = branch.body.index(setup) + 1
    _require(index < len(branch.body), "Missing injector construction loop")
    loop = branch.body[index]
    _require(isinstance(loop, ast.For) and not loop.orelse
             and _call(loop.iter, "range") and len(loop.iter.args) == 1
             and _name(loop.iter.args[0], "num_injections"),
             "Unsupported injector construction loop")
    _one((n for n in ast.walk(loop)
          if _call(n, "self.controlnet_injectors.append")), "injector append")
    _require(any(_name(n, "dec_embed_dim") for n in ast.walk(loop)),
             "Injector dimensions do not use dec_embed_dim")
    mode = _pi3_mode(branch.body[index + 1:])
    return branch.body, index + 1, mode


def _reshape(node: ast.AST, first: ast.AST, second: ast.AST) -> bool:
    return (_call(node, "multimodal_hidden.reshape") and len(node.args) == 3
            and not node.keywords and _same(node.args[0], first)
            and _same(node.args[1], second)
            and isinstance(node.args[2], ast.UnaryOp)
            and isinstance(node.args[2].op, ast.USub)
            and isinstance(node.args[2].operand, ast.Constant)
            and node.args[2].operand.value == 1)


def _decoder_anchors(model: ast.ClassDef):
    decode = _method(model, "decode")
    _require_args(decode, ["self", "hidden", "N", "H", "W", "text_embeds", "attention_mask"])
    loop = _one((n for n in decode.body if isinstance(n, ast.For)
                 and _call(n.iter, "range") and len(n.iter.args) == 1
                 and _call(n.iter.args[0], "len") and len(n.iter.args[0].args) == 1
                 and _path(n.iter.args[0].args[0]) == "self.decoder"), "main decoder loop")
    _require(_name(loop.target, "i") and not loop.orelse, "Unsupported decoder iteration")
    before = decode.body[:decode.body.index(loop)]
    batch = _one((n for n in before if _assignment(n, "B")), "batch-size derivation")
    _require(isinstance(batch.value, ast.BinOp) and isinstance(batch.value.op, ast.FloorDiv)
             and _name(batch.value.left, "BN") and _name(batch.value.right, "N"),
             "Unsupported batch-size derivation")
    size = _one((n for n in before if _assignment(n, "hw")), "post-register token count")
    _require(isinstance(size.value, ast.Subscript)
             and _path(size.value.value) == "hidden.shape"
             and isinstance(size.value.slice, ast.Constant) and size.value.slice.value == 1,
             "Unsupported post-register token count")
    product = ast.BinOp(ast.Name("B", ast.Load()), ast.Mult(), ast.Name("N", ast.Load()))
    sequence = ast.BinOp(ast.Name("N", ast.Load()), ast.Mult(), ast.Name("hw", ast.Load()))
    local = any(_reshape(n, product, ast.Name("hw", ast.Load()))
                for n in ast.walk(loop))
    global_ = any(_reshape(n, ast.Name("B", ast.Load()), sequence)
                  for n in ast.walk(loop))
    _require(local and global_, "Both per-view and cross-view hidden layouts are required")
    referring = _one((n for n in loop.body if isinstance(n, ast.If)
                      and _path(n.test) == "self.use_referring_segmentation"),
                     "decoder referring branch")
    _require(not referring.orelse, "Unsupported decoder referring else")
    fusion = _one((n for n in referring.body if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Compare) and _name(n.test.left, "i")
                   and len(n.test.ops) == len(n.test.comparators) == 1
                   and isinstance(n.test.ops[0], ast.In)
                   and _path(n.test.comparators[0]) == "self.layer_indices_map"),
                  "fusion-layer membership branch")
    _require(not fusion.orelse, "Unsupported fusion-layer else")
    residual = _one((n for n in fusion.body if _assignment(n, "x_residual")), "PWAM residual")
    _require(_call(residual.value, "fusion_module")
             and [_path(a) for a in residual.value.args] ==
             ["multimodal_hidden", "text_embeds_", "attention_mask_"]
             and not residual.value.keywords, "Unsupported PWAM call")
    offset = fusion.body.index(residual) + 1
    _require(offset + 1 < len(fusion.body), "Missing gated residual or mask projection")
    gated = fusion.body[offset]
    _require(_assignment(gated, "multimodal_hidden"), "Missing gated visual residual")
    expression = gated.value
    _require(isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
             and _name(expression.left, "multimodal_hidden")
             and isinstance(expression.right, ast.BinOp)
             and isinstance(expression.right.op, ast.Mult)
             and _call(expression.right.left, "self.res_gate")
             and len(expression.right.left.args) == 1
             and _name(expression.right.left.args[0], "x_residual")
             and not expression.right.left.keywords
             and _name(expression.right.right, "x_residual"),
             "Unsupported gated PWAM residual shape")
    projected = fusion.body[offset + 1]
    _require(_assignment(projected, "multimodal_hidden_reshaped")
             and _reshape(projected.value, product, ast.Name("hw", ast.Load())),
             "Expected mask projection immediately after PWAM residual")
    return decode.body, decode.body.index(loop), fusion.body, offset + 1


def resolve_relative_imports(tree: ast.AST) -> None:
    """Resolve imports using the supplied upstream module's fixed package."""
    package = ["mvggt", "models"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            _require(node.level <= len(package), "Relative import escapes the MVGGT package")
            parts = package[:len(package) - node.level + 1]
            if node.module:
                parts.extend(node.module.split("."))
            node.module = ".".join(parts)
            node.level = 0


def integrate_source(source: str, *, filename: str = "<upstream>") -> IntegrationResult:
    """Return transformed source, or fail before emitting any partial output."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise IntegrationError(f"Input is not valid Python: {error.msg}") from error
    first_line = tree.body[0].lineno if tree.body else 1
    preamble = "".join(source.splitlines(keepends=True)[:first_line - 1])
    model = _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MVGGT"),
                 "top-level MVGGT class")
    _require(len(model.bases) == 1 and _path(model.bases[0]) == "nn.Module"
             and not model.decorator_list and not model.keywords, "Unsupported MVGGT class shape")
    reserved = {"DepthWiseWordMemory", "RecurrentPhraseMemory", "MVGGT_M93_RPM", "word_memory"}
    _require(not any(isinstance(n, ast.Name) and n.id in reserved for n in ast.walk(tree))
             and not any(isinstance(n, ast.alias) and (n.asname or n.name) in reserved
                         for n in ast.walk(tree))
             and not any(isinstance(n, ast.ClassDef) and n.name in reserved for n in ast.walk(tree))
             and not any(isinstance(n, ast.Attribute) and n.attr in
                         {"depth_word_memory", "recurrent_phrase_memory"} for n in ast.walk(tree)),
             "Input already contains RPM symbols or conflicts with reserved names")
    init_body, init_offset, pi3_mode = _constructor_anchor(model)
    decode_body, loop_offset, fusion_body, fusion_offset = _decoder_anchors(model)
    init_body[init_offset:init_offset] = ast.parse(
        "self.depth_word_memory = DepthWiseWordMemory(dec_embed_dim, num_heads=8)\n"
        "self.recurrent_phrase_memory = RecurrentPhraseMemory(dec_embed_dim, num_heads=8)\n"
    ).body
    decode_body[loop_offset:loop_offset] = ast.parse("word_memory = None\n").body
    fusion_body[fusion_offset:fusion_offset] = ast.parse(
        "multimodal_hidden, word_memory = self.depth_word_memory("
        "multimodal_hidden, text_embeds, attention_mask, B, N, hw, word_memory)\n"
        "multimodal_hidden, word_memory = self.recurrent_phrase_memory("
        "multimodal_hidden, word_memory, attention_mask, B, N, hw)\n"
    ).body
    model.name = "MVGGT_M93_RPM"
    resolve_relative_imports(tree)
    # Keep the upstream docstring and future imports legal and in their order.
    index = int(bool(tree.body) and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str))
    while index < len(tree.body) and isinstance(tree.body[index], ast.ImportFrom) \
            and tree.body[index].module == "__future__":
        index += 1
    tree.body.insert(index, ast.ImportFrom("rpm.modules", [
        ast.alias("DepthWiseWordMemory"), ast.alias("RecurrentPhraseMemory")], 0))
    ast.fix_missing_locations(tree)
    # Preserve leading upstream comments, including any copyright/license
    # notice. AST unparsing changes formatting and removes inline comments.
    generated = preamble + ast.unparse(tree) + "\n"
    # Validate Python semantics without importing or running the resulting model.
    compile(generated, filename, "exec")
    return IntegrationResult(generated, pi3_mode)


def integrate_file(source: Path, output: Path, *, overwrite: bool = False) -> IntegrationResult:
    source, output = Path(source), Path(output)
    _require(source.is_file(), "Upstream source must be an existing Python file")
    _require(source.resolve() != output.resolve(), "Output must differ from the upstream input")
    _require(not output.is_symlink(), "Refusing a symbolic-link output")
    if output.exists() and not overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace it")
    result = integrate_source(source.read_text(encoding="utf-8-sig"), filename=str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        # Exclusive creation prevents a race from clobbering another output.
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(result.source)
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=output.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(result.source)
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True,
                        help="Local official mvggt/models/mvggt_training.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Explicit local generated model.py destination")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args(argv)
    try:
        result = integrate_file(args.upstream_source, args.output, overwrite=args.overwrite)
    except (IntegrationError, OSError, UnicodeError) as error:
        parser.exit(2, f"Integration failed: {error}\n")
    print(f"Generated local integration: {args.output}")
    print(f"Pi3 initialization: {result.pi3_initialization}")
    if result.pi3_initialization == "requires_path":
        print("The supplied upstream initializer requires pretrained_model_name_or_path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
