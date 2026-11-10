# Licensing scope and external dependencies

The MIT license in this repository covers its original RPM extension modules,
interventions, ordinary-fusion control, thin model wrappers, integration tooling,
tests, examples, and documentation. Original memory contributions were introduced
by Chuanle Song; symbol-level provenance is recorded in `SOURCE_PROVENANCE.json`.

This repository does not distribute MVGGT, Pi3, DINOv2, marepo, model weights,
tokenizers, or datasets. It is an RPM research extension, not MVGGT's official
implementation. Download external components from their authors, review their
terms, and obtain any permissions needed for your use.

## MVGGT

[Official repository](https://github.com/sosppxo/mvggt) ·
[paper](https://arxiv.org/abs/2601.06874) ·
[project](https://sosppxo.github.io/mvggt.github.io/).

MVGGT is credited to Changli Wu, Haodong Wang, Jiayi Ji, Yutian Yao, Chunsai Du,
Jihua Kang, Yanwei Fu, and Liujuan Cao. At the time of this code release, an
overall license was not found in its inspected repository root. Public access
alone does not grant this repository permission to relicense or redistribute
the complete implementation. Therefore its source is not vendored here.

The integration tool reads an existing local MVGGT source file supplied by the
user. The generated combined source includes external material and is not
covered as a whole by this repository's MIT license. Keep it with your external
dependency; do not upload it to this repository or assume unrestricted use.

## Other external components

- **marepo:** MVGGT's camera head cites code from
  [this pinned marepo version](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/LICENSE).
  Its terms include noncommercial-use restrictions, source redistribution to
  that original repository only, and conditions on distributed derived works.
  No marepo code or adapted camera-head source is included here. Consult that
  license before using or distributing an assembled MVGGT system.
- **Naver-derived upstream components:** Some MVGGT sampling, cropping and
  positional-encoding files carry CC BY-NC-SA 4.0 notices. They are not bundled.
- **DINOv2 / Meta:** Relevant upstream components carry Apache-2.0 notices.
  [DINOv2](https://github.com/facebookresearch/dinov2) is not bundled.
- **Pi3:** [Pi3](https://github.com/yyfz/Pi3#-license) distinguishes its code
  license from the CC BY-NC 4.0 terms for its model weights. Neither is bundled.
- **RoBERTa:** Obtain the text encoder and tokenizer separately; consult the
  [official model card](https://huggingface.co/FacebookAI/roberta-base).
- **PyTorch:** An installed dependency, not copied into this repository;
  see [its license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
- **ScanNet / ScanRefer / MVRefer:** Dataset access and licenses remain with
  their respective providers. No annotations, images, masks or dataset-derived
  tensors are included.

These notices describe external dependencies; they do not replace their terms
or grant additional rights to those materials.
