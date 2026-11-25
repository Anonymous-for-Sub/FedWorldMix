# FedVista

**FedVista: Robust Federated VLA Learning from Diverse Visual Worlds**

Anonymous supplementary code. This release packages the existing implementation of **VistaMix**, local VLA training, LoRA aggregation, visual-source adapters, and paired statistics.

## Installation and quick CPU check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest tests -q
```

Python 3.10 or later is required. The core mixing and aggregation tests use NumPy and do not load a VLA model. Training and rollout modules additionally use the matching PyTorch, LeRobot/SmolVLA, OpenVLA-OFT and LIBERO environments. Model weights, trajectory datasets and reconstructed scenes are supplied separately by the user.

## VistaMix and the paper

| Paper component | Implementation |
|---|---|
| VistaMix public API | `fedsplat/vistamix.py` |
| Real + 2D base and protected GS correction | `fedsplat/adaptive_augmentation.py`, `fedsplat/method.py` |
| Validity gating and perception parameter scope | `fedsplat/smolvla_local_update.py` |
| SmolVLA local adaptation | `fedsplat/smolvla_local_update.py`, `tools/run_smolvla_local_update.py` |
| OpenVLA-OFT local adaptation and evaluation | `fedsplat/openvla_oft_local_update.py`, `fedsplat/openvla_oft_rollout.py` |
| Adapter serialization and aggregation | `fedsplat/lora_updates.py`, `fedsplat/federated_adapters.py` |
| LIBERO observations and rollout | `fedsplat/libero_component_gs.py`, `fedsplat/libero_rollout.py` |
| PiPER, RoboTwin and RoboSplat adapters | `fedsplat/piper_adapter.py`, `fedsplat/robotwin_3dgs.py`, `fedsplat/robosplat_adapter.py` |
| HPT protected source updates | `tools/hpt_source_update.py`, `phaseb/` |
| Paired statistical utilities | `fedsplat/statistics.py` |

The preset coefficients are 1 for Real, 0.75 for 2D and 0.25 for GS. VistaMix first forms the Real + 2D base, projects strongly conflicting GS gradients, applies the validity weight, and caps the resulting correction relative to the base norm. Thus 0.75:0.25 specifies nominal auxiliary coefficients; it is not a fixed share of the final update. The example configs retain conflict threshold -0.02, relative norm cap 1.0 and render-coverage threshold 0.80.

```python
from fedsplat.vistamix import combine_vistamix_gradients
result = combine_vistamix_gradients(
    [1.0, 0.0], [0.0, 1.0], [-2.0, 1.0],
    gs_quality_score=0.8,
    max_gs_to_base_norm=1.0,
    projection_cosine_threshold=-0.02,
)
print(result.combined_gradient)
```

The import package remains `fedsplat`, and existing `safemix` API names and JSON schema identifiers remain readable for compatibility with saved adapters and configurations. The current paper names are FedVista (framework) and VistaMix (update rule).

## Current ten-task and ablation configurations

`configs/libero10/` contains 450 existing local-training configurations: ten tasks, three clients per task, three training repetitions, and five methods (VistaMix, NaiveJoint, Real, Real + 2D, Real + GS). `configs/ablations/` contains the existing matched sibling configurations. These preserve their numerical settings, episode partitions, budgets and seeds. Only storage roots are made portable, with the directory label `full_safemix` exposed as `vistamix`.

For the ten-task configs, provide the checkpoint at `data/smolvla_libero/` and aligned real and GS datasets under `data/libero10/task0/` through `data/libero10/task9/`, each with `canonical/` and `component_gs/`. Each GS directory includes its existing `quality_manifest.json`. For example:

```bash
python tools/run_smolvla_local_update.py \
  --config configs/libero10/vistamix/trainrep0/owner_00_task0.json \
  --out-dir outputs/task0_owner00_rep0
```

## Additional client examples

`configs/` contains the existing nine-client examples spanning LIBERO tasks 3, 5 and 9, with VistaMix and Real, Real + 2D, and Real + GS configurations. These are concrete client-training examples, not a replacement for the full ten-task evaluation protocol described in the paper. Numeric settings, owner partitions and seeds are preserved; machine-local storage roots are replaced by `./data/`.

Place the pretrained checkpoint at `data/smolvla_libero/`, aligned real and GS LeRobot datasets at `data/task{3,5,9}_pairs/{canonical,component_gs}/`, and the existing per-episode validity evidence at each GS dataset's `quality_manifest.json`. Adjust paths in the relevant JSON to your local assets. The real and GS episodes share action labels, instructions and aligned states. The loader uses the manifest's feasibility and quality fields to admit and weight GS samples.

Within the installed SmolVLA environment, one client update can be launched with:

```bash
python tools/run_smolvla_local_update.py   --config configs/vistamix/owner_00_task3.json   --out-dir outputs/owner_00_task3
```

This is a training command; the CPU test command above is sufficient to inspect the core implementation without datasets or a GPU. For OpenVLA-OFT, the module entrypoints accept `--worker-config`; provide the corresponding backbone-specific checkpoint and dataset configuration.

## Packaging

The implementation is reused from the existing project. Release edits add current-name aliases, documentation and portable example paths; they do not redesign the optimizer. No experiment outputs are regenerated by installing this package. Large checkpoints, private trajectories, runtime environments, development logs and repository history are excluded. See the unchanged accompanying supplementary PDF for the mathematical details and complete evaluation settings.

## HPT / MetaWorld implementation

The existing HPT source-composition module and its training/rollout entrypoints are included in `tools/`; their local `phaseb/` dependencies are bundled as well. The source update uses recorded multi-camera RGB observations, preserving their camera identities and action targets. It uses the HPT-specific norm-cap setting from the original implementation rather than changing it to the SmolVLA setting.

`configs/hpt/base.json` is the existing base architecture/training configuration. `tools/run_hpt_isolated_training.py` accepts explicit `--rounds`, `--seed`, `--dataset-root`, `--checkpoint`, `--output` and `--diagnostics` arguments; these select the run and override the base schedule. The preserved CLI method key `fedworldmix_hpt` selects the FedVista protected-source update. The corresponding rollout entrypoint is `tools/run_hpt_isolated_rollout.py`.

HPT additionally requires PyTorch, torchvision, einops, huggingface_hub, transformers, MetaWorld and MuJoCo, together with the original HPT assets and three-camera datasets. The existing small tensor tests are kept separately in `hpt_tests/` and can be run in that environment with `python -m pytest hpt_tests -q`. Upstream copyright and license notices are preserved in `THIRD_PARTY.md` and `licenses/`.
