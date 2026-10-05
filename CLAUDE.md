# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for distilled simulation-based inference: a conditional flow-matching "teacher" posterior is trained on SBIBM tasks, its ODE trajectories are cached, and Koopman-lifted "student" models learn to reproduce the teacher's noise→posterior map in one shot (no ODE solve at inference). Baselines (NPE, NSF, CMPE) are trained and benchmarked against the teacher and students.

There are a few versions of the Koopman distillation. There are one-shotting with and without the tensor product. The tensor product version is the main version. Optionally we can have adversarial terms, and use consistency using the original flow matching enforced.

There is no `requirements.txt`/`pyproject.toml` despite the readme. Key dependencies used by the code: `torch`, `sbibm`, `torchdiffeq`, `normflows`, `pyyaml`, `matplotlib`; optional: `bayesflow`/`keras` (CMPE `backend: "bayesflow"`), `optuna` (search scripts), `wandb`, tensorboard. The `lotka_volterra` task goes through Julia via `diffeqtorch`, so `JULIA_PROJECT`/`JULIA_LOAD_PATH` must be set (see `run_all_benchmarks.zsh`).

## Commands

Everything runs through one CLI (run from the repo root; paths in configs are relative to the CWD):

```bash
python -m koopman_sbi <command> --task two_moons        # resolves koopman_sbi/configs/tasks/two_moons.yaml
python -m koopman_sbi <command> --config path/to.yaml
```

Commands: `train-flow` (teacher), `distill-koopman` (alias `train-koopman`), `train-tensorproduct-koopman`, `train-npe`, `train-nsf`, `train-cmpe`, `evaluate`, `benchmark-suite`, `gpu-evaluation`.

- `benchmark-suite` flags: `--force-retrain`, `--plots-only` (regenerate plots from the latest saved run), `--[no-]train-flow-matching`, `--[no-]generate-teacher`. CLI overrides are written to `logs/<task>/benchmark_suite_overrides/resolved_*.yaml` and that file becomes the config path for downstream steps.
- All tasks: `./run_all_benchmarks.zsh [--from TASK]`, `./run_all_tensorproduct_koopman.zsh [--from TASK]`.
- Optuna searches: `python scripts/search_tensorproduct_koopman.py --config koopman_sbi/configs/tensorproduct_search.yaml [--trials N] [--dry-run]`; `scripts/search_teacher_context.py` uses `configs/teacher_context_search.yaml` the same way.
- Ablations: `python run_flow_simulation_ablation.py` / `run_teacher_grid_ablation.py` (configs in `configs/ablations/`).

Tests (pytest):

```bash
pytest tests
pytest tests/test_models.py::test_tensorproduct_koopman_model_shapes_and_round_trip
```

`tests/conftest.py` installs stub modules for `sbibm`, `torchdiffeq`, and `normflows` in `sys.modules` if they are not already imported, and builds a tiny CPU config (`tiny_config_path` fixture) writing logs under `tmp_path`. When the real packages are installed, the stubs are skipped, so test behaviour depends on the environment. `test_cli_smoke.py` imports the root-level `run_benchmark_suite.py`, so run pytest from the repo root.

## Architecture

**Config system** (`koopman_sbi/config.py`): YAML → nested dataclasses (`ExperimentConfig` with sections `task`, `model`, `teacher`, `training`, `evaluation`, `logging`, `benchmark_suite`). `model` and `training` each have a sub-section per model type (`flow_matching`, `koopman`, `tensorproduct_koopman`, `npe`, `nsf`, `cmpe`). Configs are layered: `configs/default.yaml` (the global base, always applied by `load_experiment_config`) ← the `base_config: <relative path>` chain ← the file itself, each deep-merged on top of the previous one (lists such as `benchmark_suite.variants` are replaced, not merged). `default.yaml` holds the shared experimental setup; task files contain only `task.name` and `task.dataset_dir`, except `gaussian_mixture.yaml`, which is experimental and overrides only the NSF/CMPE network sizes and `benchmark_suite.train_flow_matching`. The tensor-product settings tuned on it (`tensor_rank: 128`, `lambda_lat: 0.2`, 100 epochs, gradient clip 1.0) are now global; see `GAUSSIAN_MIXTURE_TENSORPRODUCT_KOOPMAN.md` for what was tried and the results. Checkpoint paths are null by default and resolve to `logs/<task>/last_model/<experiment>/best_model.pt`. Unknown keys raise a `ValueError` that names the dotted key path. When adding a config option, add the dataclass field and its value in `default.yaml`; local configs should only contain overrides. Ablation YAMLs (`configs/ablations/`) and search YAMLs have their own schemas and are not experiment configs.

**Orchestration** (`koopman_sbi/experiments/pipeline.py`): each `run_*` function loads config → seeds → `load_or_generate_dataset` → builds a model → `Trainer.fit` → reloads the best checkpoint → `evaluate_model`. Student runs (`distill_koopman`, `train_tensorproduct_koopman`) first resolve a teacher checkpoint (`teacher.checkpoint_path` → `logs/<task>/last_model/train_flow/best_model.pt` → auto-train the flow if allowed), then call `load_or_generate_teacher_trajectories`. `run_benchmark_suite` trains whatever `benchmark_suite.variants` need (one trained model can back several variants with different `sample_kwargs`, e.g. RK4 step counts), then `evaluation.benchmark_models` writes the comparison CSV, the manifest, spider/posterior plots, and the "worth it" cost analysis (training + teacher-data time vs. inference speed).

**Models** (`koopman_sbi/models/`): all subclass `BasePosteriorModel` (`compute_loss(batch) -> dict`, `sample_batch(context)`, `save`, plus a `load` classmethod). `Trainer` (`training.py`) is model-agnostic and only calls these. It runs every configured epoch (there is no early stopping) and saves a checkpoint on each improvement in val loss. The pair-trained models (flow matching, NPE/NSF via `normflows`, CMPE) consume `(theta, x)` batches from `SBIPairDataset`. The Koopman students (`koopman.py`, and `tensorproduct_koopman.py`: z_θ = E_θ(θ), z_x = E_x(x), operator K(z_x) = B + R·diag(C·z_x)·S, latent step K(z_x)·z_θ in discrete mode or exp(Δt·K(z_x))·z_θ with `use_time_dependent_consistency` (computed as a Taylor-series action, never forming the matrix), then decoder D; optional adversarial/time-dependent consistency losses) consume `TeacherTrajectoryDataset` batches of teacher (noise, endpoint[, trajectory]) triples. Networks are built from `NetworkConfig` via `networks.DenseResidualNet`.

**Tasks** (`koopman_sbi/tasks/`): `get_task(name)` returns an SBIBM task, or a local task registered in `LOCAL_TASKS`. Always use it instead of `sbibm.get_task`.
- `camera_model` (`tasks/camera_model.py`) is the GATSBI camera-model experiment, reimplemented with none of their code:
  - θ is a 28×28 EMNIST "bymerge" image. The prior is implicit, a uniform draw from the dataset (downloaded to `logs/shared/emnist`, ~2.2 GB), with the 12 test images removed.
  - x is Poisson noise followed by a σ = 3 Gaussian blur. It is written in torch; the blur matches `scipy.ndimage.gaussian_filter` exactly, and the Poisson step uses skimage's quantisation rule.
  - The 12 observations are the original experiment's held-out test pairs (`tasks/data/camera_model_observations.npz`).
  - Tasks can declare `standardization = "global"` (one mean/std over all entries; used for images) and `has_reference_posterior = False`.
- Reference-free tasks are evaluated by `image_evaluation.evaluate_image_model` (dispatched from `evaluate_model`):
  - C2ST against the teacher flow, with a torch MLP classifier; the teacher's samples are cached per checkpoint under `logs/<task>/shared/teacher_posterior_samples`.
  - Posterior-mean MSE, PSNR and SSIM against the true image, plus mean per-sample MSE, mean posterior std and 90% pixelwise coverage.
  - A posterior image grid per model.
- Image networks (`models/image_networks.py`) are selected by `NetworkConfig.type`: `ConvUNet` for the flow's vector field, and `ConvEncoder`/`ConvDecoder` for the tensor-product student's encoders and decoder. `hidden_dims` are channel widths. `downsample: false` keeps every level at full resolution (no stride-2 pooling), with the last encoder width (first decoder width) kept small so the linear map to the latent stays bounded. `projection_rank > 0` factors the flatten ↔ feature linear map through that many dimensions. The camera student uses full resolution with 2 channels at the flatten and `projection_rank: 512`, a 2048-dim latent and observation features, and `tensor_rank: 256` (11.4M parameters). A 256-dim latent capped how much of the teacher's fine-detail variation the student could track. The pull-back metric's `jacobian_type` sets how the teacher flow's Jacobian J = ∂θ/∂noise enters the endpoint loss:
  - `full` (default): exact J, O(d²) per pair, for low dimensions.
  - `diagonal`: Hutchinson estimate of diag(J·Jᵀ), the per-dimension posterior spread, from `hutchinson_probes` forward tangents.
  - `trace`: Hutchinson estimate of log|det J|, giving one isotropic weight per pair.
  - `finite_difference_trace`: the same estimate, with the probe's directional derivative taken by a forward difference (one forward pass over a doubled batch) instead of a JVP. About 3× faster for the camera U-Net on MPS, and agrees with `trace` to about 0.5%.
  - `vjp_sketch`: `hutchinson_probes` adjoint probes μ = J⁻ᵀr, integrated forward with dμ/dt = −(∇v)ᵀμ (one batched VJP per step). The loss term (1/k)·Σ(μᵢᵀe)² is an unbiased estimate of the full anisotropic pull-back metric, with O(k·d) storage per pair.
    - It runs as its own pass at `vjp_sketch_tolerance` (default 1e-3), so the trajectories keep the teacher's tolerance.
    - The tolerance can be loose: at 1e-3 the estimate changes by about 6%, against about 48% per-pair probe noise at k = 4.
    - The adjoint equation is stiff, so fixed-step RK4 is not usable for it.

  All three are computed alongside the teacher ODE and cached next to the trajectories, keyed by type and probe count. `camera_model` uses `finite_difference_trace`, because a full 784×784 J per pair is infeasible.

**Data and caching**:
- The simulated dataset is cached in `task.dataset_dir` (default `logs/<task>/shared/dataset`). Standardization (`data.Standardizer`) is fit on the training split and stored in the `DatasetBundle`.
- Teacher trajectories are cached in `logs/<task>/shared/teacher_trajectories`. The cache is invalidated by a metadata hash of the teacher checkpoint and the relevant config (`teacher.py::_trajectory_cache_metadata`). Set `teacher.generate_trajectories` / `load_cached_trajectories` to control regeneration.

**Output layout** (`paths.py`): `logs/<task>/<experiment>/<run_name or timestamp>/{checkpoints,plots,metrics,evaluation}`. The best checkpoint is also mirrored to `logs/<task>/last_model/<experiment>/best_model.pt`, which is what the `*_checkpoint_path` fields in task configs point at. Retraining overwrites these mirrors.

**Legacy/stale files**: `koopman_sbi/{koopman_flow,conditional_flow_matching,nn,train_model,train_flow_matching,pipeline}.py` are thin compatibility re-exports. `koopman_sbi/{config,pipeline_config,flow_matching_config}.yaml` are old monolithic configs, not used by the CLI. Files named `* (1).py` are stray duplicates and are not imported. Edit the canonical modules instead.

`cli.py` sets `KERAS_BACKEND=torch` and `MPLCONFIGDIR=./.cache/matplotlib` before imports. Standalone scripts do the same, so keep that when adding new entry points.

## Open issue: posterior recovery near prior boundaries

**Update:** the tuned gaussian_mixture configuration reaches C2ST ~0.57 (0.58–0.59 near the wall), against 0.654 before and 0.534 for the teacher. The adopted fixes were discrete mode, a reduced latent loss, and `tensor_rank: 128` (now global defaults), plus the pull-back endpoint metric (a version of option 3 below). The metric is `pullback_endpoint_metric`, on by default for every task in discrete mode: the endpoint error is measured through the teacher flow's Jacobian, which is computed from the vector field and cached with the trajectories. Full log: `GAUSSIAN_MIXTURE_TENSORPRODUCT_KOOPMAN.md`.

This is an active research problem. Posterior recovery is weaker on some tasks, especially gaussian_mixture when the true posterior lies near the prior boundary. In the diagnosis (raw-θ models; the gaussian_mixture student numbers came from an earlier continuous mode that used a diagonal generator instead of the tensor-product operator, since replaced):
- Teacher C2ST rose from ~0.50 in the interior to ~0.56 near the wall. The tensor-product student rose from ~0.60 to ~0.73.
- The student's per-sample endpoint error grew from ~0.09 in the interior to ~0.2 within 0.5 of the wall. For comparison, the narrow mixture component has σ = 0.1.
- gaussian_linear_uniform showed the same correlation with boundary mass, but gaussian_linear (same likelihood, unbounded prior) did not.

The suspected causes:
1. **Hard prior walls.** A truncated posterior requires a Gaussian→posterior map with an unbounded Jacobian at the wall, so the teacher's flow is stiff there and the student smooths it out. *Tried and reverted:* logit-transforming the bounded prior dimensions before standardization, so every model worked in an unconstrained space. Retrained on gaussian_mixture, it made results worse, so walls alone don't explain the gap.
2. **The student's context pathway.** E_θ and D never see the observation; context acts only through the operator K(z_x) on the latent. With a near-linear read-out, θ is then an expansion that is separable in (x, noise) with rank ≤ `tensor_rank` plus the base term. A shift of the posterior with x is cheap to represent. A shape change that depends on x (truncation at a wall) needs many modes. Not re-measured since the continuous mode was switched to the tensor-product generator. *Option, not yet implemented:* condition the decoder (and possibly the encoder) on the context while keeping the latent evolution linear.
3. **The loss doesn't reflect scale.** Endpoint MSE is averaged over all contexts in standardized θ. Rare near-wall contexts (~15% within 1 of the wall) contribute little, and a miss at the narrow-component scale (~0.017 standardized) costs almost nothing, even though C2ST penalizes it heavily. *Options, not yet implemented:* reweight near-boundary or out-of-box contexts, use an error normalized by local posterior scale, or rely more on distribution-level terms (adversarial).
