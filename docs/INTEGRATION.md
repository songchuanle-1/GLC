# Full-model integration

The standalone package needs only PyTorch. A complete system additionally needs
a user-obtained MVGGT checkout and its dependencies, data and pretrained assets.
Review [external licenses](../THIRD_PARTY_NOTICES.md) first. This repository
supplies no backbone, camera head, dataset loader, pretrained model, or tokenizer.

## Assemble locally

1. Install the external MVGGT environment following its authors' instructions.
   The audited research environment used Python 3.12 and PyTorch 2.5.1; the
   standalone CPU tests use PyTorch 2.5.1. A full model also needs the upstream
   dependencies, including its compiled/optional components where configured.
2. Install this package with `python -m pip install -e /path/to/this/repository`.
3. Run `tools/integrate_mvggt.py --upstream-source ... --output ...` as shown in the README.
   It rewrites the local model AST, inserts only the original RPM logic, and
   checks semantic anchors. An incompatible input fails instead of guessing.
4. Keep the generated file in the external model namespace at
   `mvggt/models/variants/m93_rpm/model.py`. Ensure the external checkout is
   importable, for example `export PYTHONPATH=/path/to/mvggt:$PYTHONPATH`.
5. Obtain Pi3 initialization and RoBERTa/tokenizer files from their providers.
   Configure `model.pretrained_model_name_or_path` and `model.text_model_name`.
   The former expects a directory containing `model.safetensors`.

The generated combined source keeps upstream terms and must remain outside
this repository. This public repository contains no generated backbone file.

## Hydra model targets

| Condition | Target |
|---|---|
| `rpm` | `rpm.controls.MVGGT_M93_RPM` |
| `shared_visual_read` | `rpm.controls.MVGGT_M93_SharedVisualRead` |
| `no_visual_read` | `rpm.controls.MVGGT_M93_NoVisualRead` |
| `self_only_composition` | `rpm.controls.MVGGT_M93_SelfOnlyComposition` |
| `no_token_write` | `rpm.controls.MVGGT_M93_NoTokenWrite` |
| `no_composed_write` | `rpm.controls.MVGGT_M93_NoComposedWrite` |
| `plain_fusion` | `rpm.controls.MVGGT_M93_PlainFusion` |

For a compatible upstream training entry, an example is:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 scripts/train_mvggt.py \
  train=train_mvggt_refer_lowres \
  model._target_=rpm.controls.MVGGT_M93_RPM \
  name=rpm_train work_dir=/path/to/runs \
  train_dataset.ScanNet.data_root=/path/to/multiview \
  test_dataset.ScanNet.data_root=/path/to/multiview \
  model.pretrained_model_name_or_path=/path/to/Pi3 \
  model.text_model_name=/path/to/roberta-base
```

This is an integration example. Its inherited view count, resolution, seed and
validation settings must be checked against the intended experiment. It does
not submit jobs, request a particular cluster GPU, or reproduce private guards.
The upstream entry may initialize CUDA even for `--help` or config inspection.

## Data and evaluation

- Obtain ScanNet, ScanRefer annotations and MVRefer selections from their
  providers. Do not add them to Git.
- The inspected loader expects instance masks in a sibling `scans/` directory
  beside the configured `data_root`. Follow the loader for exact layout.
- Training uses instance-to-frame mappings in `data/scene_frame_indices/`.
  Missing mappings can change sampling. This release does not supply those
  mappings or an authoritative generator.
- The optional `data/lookup.json` is needed for unique/multiple metric slices.
- The inspected upstream trainer constructs both train and validation loaders,
  including in evaluation-only mode. Both annotation/split sets may be needed.
- In that trainer, `train.resume` is an Accelerate checkpoint **directory**,
  not an arbitrary `.pth` file. Strict per-example export may additionally
  require its `resume_epoch.txt` sidecar.
- If TensorBoard logging is enabled, install `tensorboard` in that external
  environment. Optional confidence/SegFormer training needs additional packages
  and assets beyond this package's minimal PyTorch dependency.

These details describe the inspected integration, not every future MVGGT
revision. Check the external source before using a different revision.
