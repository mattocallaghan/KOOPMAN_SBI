# Tensor-product Koopman on gaussian_mixture: what was tried

Goal: improve the tensor-product Koopman student on gaussian_mixture (the posterior is a 50/50 mix of N(x, 1) and N(x, 0.01) inside a [-10, 10]² box) without affecting the other tasks, while staying fast on MPS and using only a few extra teacher calls.

## Result

| | C2ST (mean of 10 obs) | near-wall obs 1, 2, 5, 10 | interior | MMD | train time |
|---|---|---|---|---|---|
| Before (continuous mode, old gaussian_mixture config) | 0.654 | 0.674 | 0.641 | 0.0069 | 480 s |
| After, with the stretch weighting (superseded) | 0.567 | 0.580 | 0.559 | 0.0007 | 117 s + 6 s weights |
| **Current (`gaussian_mixture.yaml` + pull-back metric default)** | **0.575** | **0.590** | **0.566** | **0.0008** | 123 s + 11 s for J (0.01 s once cached) |
| Teacher (flow matching, dopri5), for reference | 0.534 | | | | |

Per-sample endpoint RMS error (θ units, on cached teacher pairs) went from 0.142 to 0.100:
- narrow component: 0.123 → 0.047
- within 1 of the wall: 0.191 → 0.106

The error that remains sits mostly in the broad (σ = 1) component, where it matters little.

All runs used 100 epochs on MPS, the same raw-θ teacher (`logs/gaussian_mixture/last_model/train_flow`), and 100k teacher pairs over 50k contexts. Each result is a single run. Seeds are fixed, but run-to-run noise is roughly ±0.01 C2ST.

## What was adopted (items 1–3 in `gaussian_mixture.yaml`; item 4 is a global option, on by default)

1. **Discrete tensor-product mode** (`use_time_dependent_consistency: false`), the same mode the other tasks use. It beat continuous mode (0.618 vs 0.654) and trains about 4× faster.
2. **Endpoint loss only** (`lambda_lat: 0`, `lambda_ae: 0`).
   - The latent loss K(z_x)·E(z) ≈ E(θ) asks for an exact linear representation of a nonlinear map, and it was the floor.
   - Removing it cut the per-sample error from 0.107 to 0.067 and MMD from 0.0033 to 0.0009.
   - The model is still z_θ = E_θ(z), z_x = E_x(x), a tensor-product linear map, then D.
3. **`tensor_rank: 128`** (from 64): 0.597 → 0.589, at no speed cost. Rank 256 gave no further gain.
4. **Pull-back endpoint metric** (`pullback_endpoint_metric: true`, `pullback_metric_epsilon: 0.3` in `default.yaml`): 0.589 → 0.575. It replaced an earlier finite-difference stretch weighting that scored the same (0.567, within noise) but was less principled. See below.

## The code change: `tensorproduct_koopman.pullback_endpoint_metric`

- **What it does:** plain MSE treats a 0.05 error the same inside the narrow spike (local scale ~0.2) and in the broad component (~1.3), but C2ST only punishes the first. The endpoint loss therefore becomes eᵀ M e with M = (J·Jᵀ + ε·I)⁻¹.
  - e = θ̂ − θ is the student's endpoint error.
  - J = dT_x/dz is the Jacobian of the teacher flow map.
  - To first order, this is the student's error measured in the teacher's noise coordinates, ‖T_x⁻¹(θ̂) − z‖², so it is relative to the local posterior scale in every direction.
- **How J is computed:** from the vector field, via the variational equation dJ/dt = ∇_θ v(t, θ_t, x)·J, J(0) = I. It is integrated together with the teacher ODE in one dopri5 solve at the teacher's tolerances: d Jacobian-vector products of v per step, no extra solves.
- **Caching:** J is stored next to the teacher trajectories (`endpoint_jacobian_{train,val}.npy`, plus `endpoint_jacobian_metadata.json` holding a copy of the trajectory cache metadata). It is reused while the trajectory cache is unchanged.
- **Regularization:** ε = (`pullback_metric_epsilon` × median σ_max(J))² keeps M bounded where J is nearly singular. M is normalized to a mean trace/d of 1 on the training split.
- **Cost:** 11 s for 100k pairs in 2-D the first time, 0.01 s once cached. Inference is unchanged. The time is reported as `endpoint_metric_time_seconds` and included in `training_plus_teacher_data_time_seconds`.
- **Scope:** on by default in `default.yaml`, so it applies to every task's discrete-mode tensor-product runs. Those tasks need retraining to pick it up, and it is untested outside gaussian_mixture. In continuous mode it is skipped with a printed notice.
- **Code:**
  - `teacher.py`: `_teacher_flow_jacobian`, `_cached_flow_jacobians`, `attach_pullback_endpoint_metric`, and an optional `endpoint_metric` on `TeacherTrajectoryDataset`.
  - `models/tensorproduct_koopman.py`: the discrete `compute_loss` uses M when the batch has a fourth element.
  - `experiments/pipeline.py`: `run_train_tensorproduct_koopman` attaches M.
- **Superseded version:** a scalar weight 1/(s² + ε) with s = σ_max(J) from finite differences (d + 1 extra solves). ε sweep: 0.03 → 0.582, 0.1 → 0.578, **0.3 → 0.574**, 1.0 → 0.589, uniform → 0.592. It was replaced because the pull-back metric uses the vector field directly and handles anisotropic posteriors.

## Pull-back metric vs stretch weighting (prototype comparison)

- **Idea:** measure the endpoint error in the teacher's noise coordinates, loss = eᵀ (J·Jᵀ + ε·I)⁻¹ e, with J = dT_x/dz.
- **How J is computed:** from the variational equation dJ/dt = ∇_θ v(t, θ_t, x)·J, J(0) = I, integrated together with the teacher ODE in one adaptive dopri5 solve. That takes d Jacobian-vector products of v per step and no extra solves.
- **ε:** (value × median σ_max(J))².
- **Validation:**
  - Endpoints from the joint solve match the cached θ to an MSE of 1.9e-9.
  - σ_max(J) agrees with the finite-difference estimate to 0.2% (median).

| Loss (prototype training loop, same model) | C2ST | near-wall | interior | MMD | weight cost (100k pairs) |
|---|---|---|---|---|---|
| stretch weight 1/(s² + ε), value 0.3 | 0.574 | 0.589 | 0.565 | 0.0010 | 6 s |
| pull-back metric, value 0.3 | 0.578 | 0.597 | 0.566 | 0.0011 | 11 s |
| pull-back metric, value 0.1 | 0.577 | 0.584 | 0.572 | 0.0017 | 11 s |

**Verdict:** a tie on gaussian_mixture. The pull-back metric was adopted on principle. That is expected: this posterior is locally round, so the largest singular value already carries the scale information. The pull-back metric only differs where the posterior is anisotropic, which gaussian_mixture does not test. The prototype is `scale_weighted.py ... pullback` in `logs/gaussian_mixture/tp_experiments/`.

## Tried and not kept (config-only; nothing left in code)

| Run | Change (on top of the previous best) | C2ST | Verdict |
|---|---|---|---|
| e1 | continuous mode, generator norm cap off | 0.639 | better than capped continuous, still worse than discrete |
| d2 | scheduler patience 5 → 20 | 0.615 | no change |
| d3 | tensor_rank 128 (with latent loss on) | 0.615 | no change while the latent loss was the floor |
| d4 | encoder/decoder width 256 → 512 | 0.625 | worse |
| d6 | lambda_ae 0 (latent loss still on) | 0.621 | no change |
| d7 | latent_dim 256 → 512 | 0.636 | worse |
| d8 | adversarial loss on (lambda_adv 0.01) | 0.611 | worse, slower |
| d10 | learning rate 1e-3 → 3e-4 | 0.599 | no change |
| d11 | scheduler patience 10 | 0.602 | no change |
| d12 | batch size 256 | 0.597 | no change, 2× slower |
| d13 | width 512 (endpoint-only loss) | 0.607 | worse |
| d15, d16 | tensor_rank 256; plus context_feature_dim 256 | 0.592, 0.596 | no gain over rank 128 |
| d17 | 200k teacher pairs | 0.601 | no gain, 2× training time |

Earlier in the project, logit-transforming the bounded θ dimensions was also tried and reverted. It placed posterior centers well but made the shapes much worse (see `CLAUDE.md`).

Diagnostics that ruled out other explanations:
- **Cached teacher targets are exact:** they match re-solved endpoints to an MSE of 1e-9.
- **The teacher map is smooth:** local stretch is at most ~2.2.
- **The model underfits:** train and val losses are equal.

## How to revert

- To return gaussian_mixture to its previous tensor-product settings, replace its `model.tensorproduct_koopman` block with:
  ```yaml
  tensorproduct_koopman:
    use_time_dependent_consistency: true
    lambda_lat: 1.0   # was lambda_phase
    lambda_end: 1.0   # was lambda_target
    lambda_ae: 1.0    # was lambda_recon (which reconstructed only the target; lambda_ae also reconstructs the noise)
    lambda_cons: 1.0
    # continuous_eigenvalue_bound: 3.0 was also set here; that option has since been removed
  ```
- To turn off the pull-back metric (globally or per task), set `pullback_endpoint_metric: false`.
- To remove it entirely, delete the pieces listed above, its two fields in `config.py` and `default.yaml`, and the cached `endpoint_jacobian_*` files.

## Reproducing

- Experiment configs, logs and helper scripts are in `logs/gaussian_mixture/tp_experiments/` (gitignored):
  - `run.sh NAME...` runs `NAME.yaml` quietly and prints one summary line each.
  - `summarize.py` produces those summary lines.
  - `diag.py RUN` gives the per-sample error breakdown.
  - `jac.py RUN OBS` measures the teacher map's stretch.
  - `scale_weighted.py` is the prototype used for the weighting sweep.
- Run outputs are in `logs/gaussian_mixture/train_tensorproduct_koopman/<run name>/`.

## Observation not acted on

The `Trainer`'s `DataLoader` over in-memory tensors dominates training time for this model: the same training loop over pre-batched tensors took 36 s instead of ~115 s. Changing the loader would affect every task and change the batch order, so it was left alone.
