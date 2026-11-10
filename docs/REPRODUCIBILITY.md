# GLC reproducibility and interpretation

The current paper title is *GLC: Grounded Language Composition for Sparse
Multi-View Referring Segmentation*. Historical RPM identifiers remain in the
`rpm` Python package, model classes, integration path, and experiment provenance.
The `MVGGT_GLC*` model targets are direct aliases of existing classes. This naming
update does not reconstruct models, rename checkpoint keys, or change memory
computation. Keep historical identifiers when matching an existing run.

## What the code preserves

The released word/phrase memory classes and control functions retain their
research implementation computation. The integrator preserves the supplied
backbone's code and inserts the two modules at its existing PWAM fusion sites.
It does not substitute a new backbone or silently change its optimizer.

The memory classes start with zero output projections. A newly initialized
module can therefore preserve its visual input while internal state updates
already execute. Nonzero behavior and gradient tests explicitly activate those
projections where needed; a zero gradient is not evidence that a parameter was
omitted from an optimizer.

## Conditions are implementations, not results

The repository includes all six intended controls, including unfinished or
unrun conditions. It makes no claim that every condition completed a full
training protocol. No private logs, selected checkpoints, result tables,
manuscript, annotation records or training weights are distributed here.

The original campaign uses one fixed initialization/data tuple per condition;
that does not estimate multi-seed uncertainty or establish statistical
significance. Compare equal training budgets and report unfavorable outcomes.
Separate the validation-selected checkpoint from the final checkpoint.

## Matching a full experiment

Before claiming reproduction, preserve and report:

- Dataset/sampling configuration, view count, resolution, text tokenizer and
  exact validation examples.
- Pretrained assets, parameter initialization and all seed/RNG settings.
- Mixed precision, optimizer parameter membership, scheduler and real updates.
- Validation frequency, selection metric, final epoch and checkpoint loading.
- Operational failures and any incomplete protocol, separately from scientific
  scores. A partial run is not comparable to a complete training budget.

The historical recipe's name-based optimizer filter omits parameters with
`.encoder.` in their names, including a pre-existing omitted trainable
population. The new memory/control names intentionally avoid that substring.
Changing the historical omitted population creates a different recipe; verify
the actual optimizer rather than assuming all `requires_grad` parameters are
updated. This release does not silently repair that filter.

No shorter/final-only validation schedule is claimed trajectory-equivalent.
If changing validation, check RNG, next batch, modes, parameters/buffers,
included and omitted gradients, optimizer and scheduler state.

## Control semantics

`self_only_composition` removes explicit cross-token exchange in the added
pair module; contextual RoBERTa embeddings still carry token context. Destination
and source projections become redundant roles. Equal parameter inventory does
not imply equal effective capacity.

Zero-read removes only the added visual read; PWAM remains. Shared-read averages
only valid-token visual evidence. Both write controls execute their computation
and then gate the specified residual. The plain control is near parameter
matched (512 fewer scalars at width1024), without FLOP, latency or memory matching.

CPU tests establish module-level behavior and integration structure only.
They do not validate GPU numerical equivalence, full-model quality, timing,
or the scientific outcome of a training condition.
