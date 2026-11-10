# Grounded Language States for Sparse Multi-View Referring Segmentation

Research code for **Recurrent Phrase Memory (RPM)**, the **read → compose → write**
language-state extension in *Grounded Language States for Sparse Multi-View
Referring Segmentation*, built around an external MVGGT backbone.

This release contains the method implementation, six controls, CPU tests, and
a local integration tool. The backbone, datasets, pretrained assets, trained
checkpoints, manuscript, and internal experiment infrastructure are not included.
See [licensing scope](THIRD_PARTY_NOTICES.md) before assembling a full system.

## Method

At each fusion depth, `DepthWiseWordMemory` reads visual evidence into a hidden
state for each valid language token using a GRU. `RecurrentPhraseMemory` forms
ordered token-pair messages, updates those states, and writes composed language
information back to visual tokens. Both modules are shared across fusion depth.
The word and phrase visual-output projections start at zero.

```text
visual tokens ──read──► token states ──compose──► phrase states
      ▲                      │                       │
      └───── token write ────┘──── composed write ───┘
```

The standalone modules are extracted from the research implementation without
changing their computation. The integration tool inserts them after the original
PWAM gated residual at the existing fusion positions; it retains the external
model's decoder, masking, checkpointing and output heads.

## Install and run a small example

Python 3.10+ and PyTorch 2.5+ are required. Install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/) first, then:

```bash
git clone https://github.com/songchuanle-1/grounded-language-states-for-sparse-multi-view-referring-segmentation.git
cd grounded-language-states-for-sparse-multi-view-referring-segmentation
python -m pip install -e .
python examples/module_demo.py
python -m unittest discover -s tests -v
```

The example and module tests use small CPU tensors and need no backbone,
checkpoint, dataset, tokenizer or GPU. Full multiview training needs the external
backbone's dependencies and assets.

## Available controls

| Condition | Intervention |
|---|---|
| `rpm` | Original token memory, ordered pair composition and both visual writes. |
| `shared_visual_read` | Average only valid-token visual-read evidence and broadcast it before normalization/GRU. |
| `no_visual_read` | Zero only the added visual read; original PWAM remains. |
| `self_only_composition` | Replace each source-token contribution by the receiver's own contribution; contextual language embeddings remain. |
| `no_token_write` | Compute the original token route and gate its visual residual to zero. |
| `no_composed_write` | Compute the original composed route and gate its visual residual to zero. |
| `plain_fusion` | Ordinary attention/FFN replacement without GRU or explicit pair MLP. |

These controls preserve stored parameters where specified, not necessarily
active capacity: gated routes can receive zero gradients. At width 1024, the
plain modules have 26,252,800 parameters, versus RPM's 26,253,312: 512 fewer
(about 0.00195%). This is a near parameter match, **not a FLOP or runtime match**.

## Integrate with a separately obtained MVGGT

Obtain [MVGGT](https://github.com/sosppxo/mvggt) from its authors, review its
applicable terms, and install its dependencies. This tool requires the matching
training-model structure and fails if expected anchors differ. It does not fetch
or redistribute the external source.

```bash
python tools/integrate_mvggt.py \
  --upstream-source /path/to/mvggt/mvggt/models/mvggt_training.py \
  --output /path/to/mvggt/mvggt/models/variants/m93_rpm/model.py
```

Keep that output inside your external checkout, outside this public repository.
Install this RPM package and put the external checkout on Python's import path.
After integration, model targets are available from `rpm.controls`:

```python
from rpm.controls import MVGGT_M93_RPM, MVGGT_M93_SelfOnlyComposition
```

The full model still needs the external Pi3 initializer, RoBERTa assets, and the
backbone's configuration. See [full-model setup](docs/INTEGRATION.md) for targets,
example commands and data requirements. The tool does not run training or modify
any existing file unless an explicit overwrite option is provided.

## Reproducibility scope

The release provides method code; it is not an archive of completed paper
experiments. Implementing a control does not imply its full scientific protocol
has completed. Do not infer seed consistency, significance, a persistence
advantage, or a monotonic ordering of controls from these classes or CPU tests.

Full paper comparisons require the original data/initialization, optimizer
membership, precision, training budget, validation cohort and selection rule.
The provided integration instructions are examples, not a claim that a fresh
environment reproduces accepted runs. See [reproducibility notes](docs/REPRODUCIBILITY.md).

## Repository layout

```text
rpm/modules.py          Original word and phrase memory
rpm/interventions.py    Five targeted interventions
rpm/fusion.py           Ordinary attention/FFN capacity control
rpm/controls.py         Full-model wrappers; external MVGGT required
tools/integrate_mvggt.py Local source integration, without vendored backbone
examples/               Small CPU example and target mapping
tests/                  Module, gradient and integration checks
```

## Credit and license

The original extension is released under [MIT](LICENSE). That grant does not
relicense MVGGT or its dependencies. This repository is independent of the
[official MVGGT implementation](https://github.com/sosppxo/mvggt); credit for the
backbone and benchmark belongs to its authors. Consult
[third-party notices](THIRD_PARTY_NOTICES.md) for external restrictions.
