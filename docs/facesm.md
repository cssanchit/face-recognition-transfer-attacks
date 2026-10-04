# FaceSM integration and reproduction

## Shared implementation

`core/facesm_objective.py` ports `compute_embedding` and `attack_loss_sm` from the supplied `paper/core/facesm_attack_core.py`. Each view is L2-normalized before averaging and renormalization. The source and target references use the same fusion.

For target cosine `c_t`, source cosine `c_s`, and weight `lambda`, the maximized scores are:

- Impersonation: `c_t - lambda * c_s`.
- Dodging: `(1 - c_t) + lambda * (1 - c_s)`.

A per-pair model wrapper performs mirror fusion and holds the fixed source embedding. All gradient objectives in the shared attack implementations use the same loss helper, including neighbor, transformation, and DECOWA inner-loop objectives. Vanilla calls retain their original cosine score. No global loss monkey-patching or model-weight mutation is used.

The integration retains each existing attack's optimizer, iteration count, and perturbation settings. It does not promise exact reproduction of the paper: the contributed implementations and defaults can differ from the paper's experiment configuration. Extra backbones are extensions, not new measured results. DYNAMIC_MORPH remains vanilla-only due to its different perturbation reference, and the incomplete IDAA registry entry has been disabled.

## Original paper materials

`paper/` preserves the 40 files supplied in `facesmanonymous-7427.zip`, including original documentation and attribution. Historical references to anonymity describe that review snapshot. The public maintained entry point is the top-level generation script.

The archive is not a complete one-command reproduction environment. In particular:

- The historical core imports `adv_output_cleanup`, which was not included.
- The ablation, lambda-sweep, and evaluator workflows need `ir152_model` and IR-152 weights, which were not included.
- Some historical scripts assume local directory layouts and input/output filenames. Inspect and configure them before use; some cleanup/resume options can delete generated files.
- RobFR and Sibling-Attack validation require their external frameworks, dependencies, checkpoints, and datasets. Follow their original documentation and licenses.
- The historical requirements list omits TensorFlow despite TensorFlow imports. The top-level requirements provide dependencies for the maintained runner; historical reproduction can need additional dependencies from `paper/requirements.txt`.

Use `paper/docs/manuscript_result_map.md` and `paper/results_summary/` to inspect the supplied aggregate results. Do not interpret successful smoke tests as reproducing those results.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use a small synthetic TensorFlow model without face data or downloaded weights to check the objective equations, input gradients, fused views, transformed batches, perturbation bounds, and isolation of vanilla runs.

## Attribution and reuse

FaceSM authors: Sanchit Gupta, Vishakha Agrawal, Pratishtha Jaiswal, Ananya Jain. Accepted for presentation at ICISS 2026. Original attack authors and implementation contributors remain credited in the root README, CONTRIBUTORS.md, and attack-specific documentation.

Neither the supplied vanilla repository nor the FaceSM archive included a root license. This integration does not invent a license or override third-party terms. A maintainer should select an appropriate license for original contributions and retain applicable third-party licenses before representing all code as permissively licensed.
