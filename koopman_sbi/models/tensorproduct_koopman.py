from __future__ import annotations

from typing import Any, Dict, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_sbi.config import AdversarialConfig, NetworkConfig, TensorProductKoopmanModelConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.koopman import KoopmanDiscriminator
from koopman_sbi.models.image_networks import ConvDecoder, ConvEncoder
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


# Taylor order per substep of the matrix-exponential action; substeps keep ||dt * G|| <= 1 each,
# so the truncation error per substep is below 1/13! (~1.6e-10).
_EXPM_MAX_SUBSTEP_NORM = 2.0  # larger substeps lose float32 digits to cancellation between Taylor terms
_EXPM_MAX_SUBSTEPS = 64
_EXPM_MAX_ORDER = 30


def _taylor_schedule(norm: float, dtype: torch.dtype) -> tuple[int, int]:
    """(substeps, order) minimising operator applications for exp(A) with ||A|| <= norm.

    Truncation error per substep is bounded by rho^(order+1) / (order+1)! with rho = norm / substeps; it is kept
    below the working precision (float32 ~1e-7), so the series is exact to that precision without spending
    applications on accuracy the dtype cannot represent.
    """
    tolerance = 1e-7 if dtype in (torch.float32, torch.float16, torch.bfloat16) else 1e-15
    best = None
    for substeps in range(max(1, math.ceil(norm / _EXPM_MAX_SUBSTEP_NORM)), _EXPM_MAX_SUBSTEPS + 1):
        rho = norm / substeps
        order, bound = 1, rho * rho / 2.0
        while bound > tolerance and order < _EXPM_MAX_ORDER:
            order += 1
            bound *= rho / (order + 1)
        cost = substeps * order
        if best is None or cost < best[0]:
            best = (cost, substeps, order)
        if bound <= tolerance and substeps * order > best[0] + order:
            break
    return best[1], best[2]


def _network_config_to_dict(config: NetworkConfig) -> Dict[str, object]:
    return {
        "hidden_dims": config.hidden_dims,
        "activation": config.activation,
        "batch_norm": config.batch_norm,
        "dropout": config.dropout,
        "theta_with_glu": config.theta_with_glu,
        "context_with_glu": config.context_with_glu,
        "type": config.type,
        "downsample": config.downsample,
        "projection_rank": config.projection_rank,
    }


class TensorProductKoopmanFlow(BasePosteriorModel):
    """One-shot Koopman distillation model with a low-rank bilinear context operator.

    z_theta = E_theta(theta) and z_x = E_x(x) are combined through the tensor-product operator
    K(z_x) = B + R diag(C z_x) S (base operator plus a rank-`tensor_rank` context term). In discrete mode
    the latent is propagated as K(z_x) z_theta; with `use_time_dependent_consistency` the same K(z_x) is the
    generator of a linear latent ODE and the latent is propagated by exp(dt K(z_x)) z_theta. The result is
    decoded by D.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: TensorProductKoopmanModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        self.teacher_model = None
        if int(model_config.latent_dim) < 1:
            raise ValueError("tensorproduct_koopman.latent_dim must be positive.")
        if int(model_config.context_feature_dim) < 1:
            raise ValueError("tensorproduct_koopman.context_feature_dim must be positive.")
        if int(model_config.tensor_rank) < 1:
            raise ValueError("tensorproduct_koopman.tensor_rank must be positive.")

        state_cfg = model_config.state_network
        context_cfg = model_config.context_network
        decoder_cfg = model_config.decoder_network
        latent_dim = int(model_config.latent_dim)
        context_feature_dim = int(model_config.context_feature_dim)
        tensor_rank = int(model_config.tensor_rank)
        state_input_dim = input_dim + 1 if model_config.use_time_dependent_consistency else input_dim

        if state_cfg.type == "ConvEncoder":
            self.state_encoder = ConvEncoder(
                state_input_dim, latent_dim, state_cfg.hidden_dims, num_pixels=input_dim, downsample=state_cfg.downsample,
                projection_rank=state_cfg.projection_rank,
            )
        elif state_cfg.type == "DenseResidualNet":
            self.state_encoder = DenseResidualNet(
                input_dim=state_input_dim,
                output_dim=latent_dim,
                hidden_dims=state_cfg.hidden_dims,
                activation=state_cfg.activation,
                batch_norm=state_cfg.batch_norm,
                dropout=state_cfg.dropout,
                theta_dim=state_input_dim,
                theta_with_glu=state_cfg.theta_with_glu,
            )
        else:
            raise ValueError(f"Unsupported tensorproduct_koopman network type: {state_cfg.type!r}")
        if context_cfg.type == "ConvEncoder":
            self.context_encoder = ConvEncoder(
                context_dim,
                context_feature_dim,
                context_cfg.hidden_dims,
                num_pixels=context_dim,
                downsample=context_cfg.downsample,
                projection_rank=context_cfg.projection_rank,
            )
        elif context_cfg.type == "DenseResidualNet":
            self.context_encoder = DenseResidualNet(
                input_dim=context_dim,
                output_dim=context_feature_dim,
                hidden_dims=context_cfg.hidden_dims,
                activation=context_cfg.activation,
                batch_norm=context_cfg.batch_norm,
                dropout=context_cfg.dropout,
                context_dim=context_dim,
                context_with_glu=context_cfg.context_with_glu,
            )
        else:
            raise ValueError(f"Unsupported tensorproduct_koopman network type: {context_cfg.type!r}")
        if decoder_cfg.type == "ConvDecoder":
            self.decoder = ConvDecoder(
                latent_dim,
                input_dim,
                decoder_cfg.hidden_dims,
                downsample=decoder_cfg.downsample,
                projection_rank=decoder_cfg.projection_rank,
            )
        elif decoder_cfg.type == "DenseResidualNet":
            self.decoder = DenseResidualNet(
                input_dim=latent_dim,
                output_dim=input_dim,
                hidden_dims=decoder_cfg.hidden_dims,
                activation=decoder_cfg.activation,
                batch_norm=decoder_cfg.batch_norm,
                dropout=decoder_cfg.dropout,
                theta_dim=latent_dim,
                theta_with_glu=decoder_cfg.theta_with_glu,
            )
        else:
            raise ValueError(f"Unsupported tensorproduct_koopman network type: {decoder_cfg.type!r}")
        self.base_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        self.state_factor = nn.Linear(latent_dim, tensor_rank, bias=False)
        self.context_factor = nn.Linear(context_feature_dim, tensor_rank, bias=False)
        self.factor_reconstruction = nn.Linear(tensor_rank, latent_dim, bias=False)
        self.discriminator = (
            KoopmanDiscriminator(theta_dim=input_dim, context_dim=context_dim, config=model_config.adversarial)
            if model_config.adversarial.enabled
            else None
        )

    def generator_parameters(self):
        modules = [
            self.state_encoder,
            self.context_encoder,
            self.decoder,
            self.base_operator,
            self.state_factor,
            self.context_factor,
            self.factor_reconstruction,
        ]
        for module in modules:
            yield from module.parameters()

    def discriminator_parameters(self):
        if self.discriminator is None:
            return
        yield from self.discriminator.parameters()

    def set_teacher_model(self, teacher_model: nn.Module) -> None:
        self.__dict__["teacher_model"] = teacher_model
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    def _time_column(self, time: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(time):
            time = torch.full((state.shape[0], 1), float(time), device=state.device, dtype=state.dtype)
        else:
            time = time.to(device=state.device, dtype=state.dtype)
            if time.dim() == 0:
                time = time.expand(state.shape[0]).unsqueeze(1)
            elif time.dim() == 1:
                time = time.unsqueeze(1)
        return time

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.model_config.use_time_dependent_consistency:
            state = torch.cat([self._time_column(0.0, state), state], dim=-1)
        return self.state_encoder(state)

    def encode_observable(self, time: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        if self.model_config.use_time_dependent_consistency:
            state = torch.cat([self._time_column(time, state), state], dim=-1)
        return self.state_encoder(state)

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def _context_coefficients(self, context_feature: torch.Tensor) -> torch.Tensor:
        return self.context_factor(context_feature)

    def _apply_operator(self, latent_state: torch.Tensor, context_coefficients: torch.Tensor) -> torch.Tensor:
        """K(z_x) z = B z + R ((S z) * (C z_x)), without forming the per-sample matrix."""
        bilinear_update = self.factor_reconstruction(self.state_factor(latent_state) * context_coefficients)
        return self.base_operator(latent_state) + bilinear_update

    def evolve_latent(self, latent_state: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        return self._apply_operator(latent_state, self._context_coefficients(context_feature))

    def _apply_operator_transpose(self, latent_state: torch.Tensor, context_coefficients: torch.Tensor) -> torch.Tensor:
        """K(z_x)^T w = B^T w + S^T ((R^T w) * (C z_x))."""
        low_rank = (latent_state @ self.factor_reconstruction.weight) * context_coefficients
        return latent_state @ self.base_operator.weight + low_rank @ self.state_factor.weight

    def _operator_norm_estimate(self, context_coefficients: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        """Per-sample ||K(z_x)||_2 by power iteration on K^T K (detached); used only to choose Taylor substeps."""
        with torch.no_grad():
            coefficients = context_coefficients.detach()
            vector = F.normalize(
                torch.randn(
                    coefficients.shape[0],
                    self.base_operator.weight.shape[1],
                    device=coefficients.device,
                    dtype=coefficients.dtype,
                ),
                dim=-1,
            )
            for _ in range(iterations):
                vector = F.normalize(
                    self._apply_operator_transpose(self._apply_operator(vector, coefficients), coefficients),
                    dim=-1,
                )
            return self._apply_operator(vector, coefficients).norm(dim=-1)

    @staticmethod
    def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
        if torch.isfinite(tensor).all():
            return
        detached = tensor.detach()
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            raise FloatingPointError(f"Non-finite tensor in tensorproduct Koopman: {name}; no finite values.")
        raise FloatingPointError(
            "Non-finite tensor in tensorproduct Koopman: "
            f"{name}; finite_min={finite.min().item():.6g} finite_max={finite.max().item():.6g}"
        )

    def latent_generator_rhs(self, latent_state: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        """dz/dt of the latent ODE: the tensor-product operator applied as the generator."""
        return self._apply_operator(latent_state, self._context_coefficients(context_feature))

    def _generator_adversarial_loss(
        self,
        predicted_theta: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.discriminator is None:
            return predicted_theta.new_zeros(())
        fake_logits = self.discriminator(predicted_theta, context)
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    def _discriminator_loss(
        self,
        theta_target: torch.Tensor,
        predicted_theta: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.discriminator is None:
            return predicted_theta.new_zeros(())
        real_logits = self.discriminator(theta_target, context)
        fake_logits = self.discriminator(predicted_theta.detach(), context)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real_loss + fake_loss)

    def evolve_latent_continuous(
        self,
        latent_state: torch.Tensor,
        context_feature: torch.Tensor,
        delta_time: torch.Tensor | float,
    ) -> torch.Tensor:
        """exp(dt K(z_x)) z via a scaled Taylor series of the operator action (matrix-vector products only)."""
        context_coefficients = self._context_coefficients(context_feature)
        delta_column = self._time_column(delta_time, latent_state).clamp_min(0.0)
        # The 10% margin covers power iteration slightly underestimating the norm.
        norm_estimate = 1.1 * self._operator_norm_estimate(context_coefficients)
        max_norm = (norm_estimate * delta_column.squeeze(-1).detach()).max().item()
        num_substeps, taylor_order = _taylor_schedule(max_norm, latent_state.dtype)
        step = delta_column / num_substeps
        evolved = latent_state
        for _ in range(num_substeps):
            term = evolved
            for order in range(1, taylor_order + 1):
                term = self._apply_operator(term, context_coefficients) * (step / order)
                evolved = evolved + term
        self._raise_if_nonfinite("continuous_evolved_latent", evolved)
        return evolved

    def forward(self, initial_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latent_initial = self.encode_state(initial_state)
        context_feature = self.encode_context(context)
        latent_predicted = self.evolve_latent(latent_initial, context_feature)
        return self.decode_state(latent_predicted)

    def compute_loss(self, batch: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del kwargs
        if self.model_config.use_time_dependent_consistency and len(batch) == 6:
            return self._compute_full_trajectory_loss(batch)
        if self.model_config.use_time_dependent_consistency:
            return self._compute_time_dependent_loss(batch, include_consistency=False)

        initial_state, target_state, context = batch[:3]
        endpoint_metric = batch[3] if len(batch) == 4 else None
        latent_initial = self.encode_state(initial_state)
        latent_target = self.encode_state(target_state)
        context_feature = self.encode_context(context)
        latent_predicted = self.evolve_latent(latent_initial, context_feature)

        reconstructed_target = self.decode_state(latent_target)
        predicted_target = self.decode_state(latent_predicted)

        # Autoencoding only at the endpoint, in sample space: D(E(theta)) = theta.
        ae_loss = nn.MSELoss()(reconstructed_target, target_state)
        latent_loss = nn.MSELoss()(latent_predicted, latent_target)
        endpoint_loss = self._endpoint_loss(predicted_target, target_state, endpoint_metric)
        total_loss = (
            float(self.model_config.lambda_ae) * ae_loss
            + float(self.model_config.lambda_lat) * latent_loss
            + float(self.model_config.lambda_end) * endpoint_loss
        )
        metrics = {
            "total_loss": total_loss,
            "autoencoder_loss": ae_loss,
            "latent_loss": latent_loss,
            "endpoint_loss": endpoint_loss,
        }
        if self.discriminator is not None:
            adversarial_loss = self._generator_adversarial_loss(predicted_target, context)
            metrics["total_loss"] = total_loss + float(self.model_config.adversarial.lambda_adv) * adversarial_loss
            metrics["generator_adversarial_loss"] = adversarial_loss
        return metrics

    @staticmethod
    def _endpoint_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        endpoint_metric: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Plain MSE, or e^T M e / d with the per-pair pull-back metric (see teacher.attach_pullback_endpoint_metric).

        M is a (d, d) matrix ("full"), a per-dimension weight vector ("diagonal"), a scalar ("trace") per pair, or
        for "vjp_sketch" low-rank factors U (d, k) with M = I + U U^T.
        """
        if endpoint_metric is None:
            return nn.MSELoss()(predicted, target)
        error = predicted - target
        if isinstance(endpoint_metric, dict):
            projections = torch.einsum("bi,bik->bk", error, endpoint_metric["low_rank_factors"])
            return (error.pow(2).sum(-1) + projections.pow(2).sum(-1)).mean() / error.shape[1]
        if endpoint_metric.dim() == 3:
            return torch.einsum("bi,bij,bj->b", error, endpoint_metric, error).mean() / error.shape[1]
        if endpoint_metric.dim() == 2:
            return (endpoint_metric * error.pow(2)).mean()
        return (endpoint_metric * error.pow(2).mean(dim=1)).mean()

    def _compute_time_dependent_loss(
        self,
        batch: Any,
        include_consistency: bool,
    ) -> Dict[str, torch.Tensor]:
        initial_state, target_state, context = batch[:3]
        endpoint_metric = batch[3] if len(batch) == 4 else None
        context_feature = self.encode_context(context)
        latent_initial = self.encode_observable(0.0, initial_state)
        latent_target = self.encode_observable(1.0, target_state)
        latent_predicted = self.evolve_latent_continuous(
            latent_initial,
            context_feature,
            delta_time=float(self.model_config.continuous_final_time),
        )
        predicted_target = self.decode_state(latent_predicted)
        reconstructed_target = self.decode_state(latent_target)

        # Same terms as the one-shot loss (autoencoder at the endpoint, latent, endpoint), plus consistency.
        ae_loss = nn.MSELoss()(reconstructed_target, target_state)
        latent_loss = nn.MSELoss()(latent_predicted, latent_target)
        endpoint_loss = self._endpoint_loss(predicted_target, target_state, endpoint_metric)
        consistency_loss = endpoint_loss.new_zeros(())
        if include_consistency and float(self.model_config.lambda_cons) != 0.0:
            consistency_loss = self._compute_teacher_velocity_consistency(
                initial_state,
                target_state,
                context,
                context_feature,
            )
        return self._weighted_loss_metrics(ae_loss, latent_loss, endpoint_loss, consistency_loss, predicted_target, context)

    def _weighted_loss_metrics(
        self,
        ae_loss: torch.Tensor,
        latent_loss: torch.Tensor,
        endpoint_loss: torch.Tensor,
        consistency_loss: torch.Tensor,
        predicted_target: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Continuous-mode total: lambda_ae/lambda_lat/lambda_end as in the one-shot loss, plus lambda_cons."""
        total_loss = (
            float(self.model_config.lambda_ae) * ae_loss
            + float(self.model_config.lambda_lat) * latent_loss
            + float(self.model_config.lambda_end) * endpoint_loss
            + float(self.model_config.lambda_cons) * consistency_loss
        )
        metrics = {
            "total_loss": total_loss,
            "autoencoder_loss": ae_loss,
            "latent_loss": latent_loss,
            "endpoint_loss": endpoint_loss,
            "consistency_loss": consistency_loss,
        }
        if self.discriminator is not None:
            adversarial_loss = self._generator_adversarial_loss(predicted_target, context)
            metrics["total_loss"] = total_loss + float(self.model_config.adversarial.lambda_adv) * adversarial_loss
            metrics["generator_adversarial_loss"] = adversarial_loss
        return metrics

    def _compute_teacher_velocity_consistency(
        self,
        initial_state: torch.Tensor,
        target_state: torch.Tensor,
        context: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher_model is None:
            raise RuntimeError(
                "TensorProductKoopmanFlow time-dependent consistency mode requires a teacher model for training."
            )
        batch_size = target_state.shape[0]
        time = torch.rand(batch_size, device=target_state.device, dtype=target_state.dtype)
        sigma_min = float(getattr(self.teacher_model.model_config, "sigma_min", 0.0))
        path_state = (1.0 - (1.0 - sigma_min) * time)[:, None] * initial_state + time[:, None] * target_state
        with torch.no_grad():
            teacher_velocity = self.teacher_model.forward(time, path_state, context).detach()

        def encode_fn(time_input: torch.Tensor, state_input: torch.Tensor) -> torch.Tensor:
            return self.encode_observable(time_input, state_input)

        _, encoded_directional_derivative = torch.autograd.functional.jvp(
            encode_fn,
            (time, path_state),
            (torch.ones_like(time), teacher_velocity),
            create_graph=True,
        )
        latent_state = self.encode_observable(time, path_state)
        generator_direction = self.latent_generator_rhs(latent_state, context_feature)
        return nn.MSELoss()(generator_direction, encoded_directional_derivative)

    def train_batch(
        self,
        batch: Any,
        optimizer,
        discriminator_optimizer=None,
        gradient_clip_norm: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        discriminator_loss = None
        if discriminator_optimizer is not None and self.discriminator is not None:
            _, theta_target, context = batch[:3]
            with torch.no_grad():
                predicted_theta = self.sample_batch(context, initial_noise=batch[0])
            # sample_batch switches to eval mode; restore train mode for the generator update.
            self.train()
            discriminator_optimizer.zero_grad()
            discriminator_loss = self._discriminator_loss(theta_target, predicted_theta, context)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        if self.model_config.use_time_dependent_consistency:
            loss_dict = self._compute_time_dependent_loss(batch, include_consistency=True)
        else:
            loss_dict = self.compute_loss(batch)
        if discriminator_loss is not None:
            loss_dict["discriminator_loss"] = discriminator_loss
        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.generator_parameters()),
                gradient_clip_norm,
            )
        else:
            for name, parameter in self.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient in tensorproduct Koopman parameter: {name}")
        optimizer.step()
        return {
            key: value.detach()
            for key, value in loss_dict.items()
            if key != "predicted_theta"
        }

    def _compute_full_trajectory_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        initial_state, target_state, context, time_grid, path, path_mask = batch
        del initial_state
        context_feature = self.encode_context(context)
        batch_size, num_steps, state_dim = path.shape
        flat_path = path.reshape(batch_size * num_steps, state_dim)
        flat_time = time_grid.reshape(batch_size * num_steps)
        flat_latent = self.encode_observable(flat_time, flat_path)
        latent_path = flat_latent.reshape(batch_size, num_steps, -1)

        path_mask = path_mask.bool()
        valid_counts = path_mask.long().sum(dim=1).clamp_min(1)
        endpoint_index = valid_counts - 1
        gather_index = endpoint_index[:, None, None].expand(-1, 1, latent_path.shape[-1])
        target_latent_from_path = latent_path.gather(dim=1, index=gather_index).squeeze(1)
        start_time = time_grid[:, 0]
        end_time = time_grid.gather(dim=1, index=endpoint_index[:, None]).squeeze(1)
        delta_time = end_time - start_time
        latent_initial = latent_path[:, 0]
        latent_predicted = self.evolve_latent_continuous(
            latent_initial,
            context_feature,
            delta_time=delta_time,
        )
        predicted_target = self.decode_state(latent_predicted)

        segment_mask = (path_mask[:, :-1] & path_mask[:, 1:]).reshape(-1)
        segment_start_latent = latent_path[:, :-1].reshape(-1, latent_path.shape[-1])
        segment_target_latent = latent_path[:, 1:].reshape(-1, latent_path.shape[-1])
        segment_context = context_feature[:, None, :].expand(-1, num_steps - 1, -1).reshape(
            -1,
            context_feature.shape[-1],
        )
        segment_delta_time = (time_grid[:, 1:] - time_grid[:, :-1]).reshape(-1)
        segment_predicted = self.evolve_latent_continuous(
            segment_start_latent,
            segment_context,
            delta_time=segment_delta_time,
        )

        def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            while mask.dim() < prediction.dim():
                mask = mask.unsqueeze(-1)
            squared_error = (prediction - target).pow(2) * mask.to(prediction.dtype)
            denominator = mask.to(prediction.dtype).sum().clamp_min(1.0) * prediction.shape[-1]
            return squared_error.sum() / denominator

        # Autoencoding only at the endpoint, in sample space.
        ae_loss = nn.MSELoss()(self.decode_state(target_latent_from_path), target_state)
        latent_loss = nn.MSELoss()(latent_predicted, target_latent_from_path)
        endpoint_loss = nn.MSELoss()(predicted_target, target_state)
        consistency_loss = masked_mse(segment_predicted, segment_target_latent, segment_mask)
        return self._weighted_loss_metrics(ae_loss, latent_loss, endpoint_loss, consistency_loss, predicted_target, context)

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            context = move_tensor_to_device(context, self.device)
            batch_size = context.shape[0]
            initial_state = (
                initial_noise
                if initial_noise is not None
                else torch.randn(batch_size, self.input_dim, device=self.device)
            )
            initial_state = move_tensor_to_device(initial_state, self.device)
            if self.model_config.use_time_dependent_consistency:
                context_feature = self.encode_context(context)
                latent_initial = self.encode_observable(0.0, initial_state)
                final_time = float(self.model_config.continuous_final_time)
                latent_final = self.evolve_latent_continuous(
                    latent_initial,
                    context_feature,
                    delta_time=final_time,
                )
                return self.decode_state(latent_final)
            return self.forward(initial_state, context)

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "latent_dim": self.model_config.latent_dim,
                    "context_feature_dim": self.model_config.context_feature_dim,
                    "tensor_rank": self.model_config.tensor_rank,
                    "lambda_ae": self.model_config.lambda_ae,
                    "lambda_lat": self.model_config.lambda_lat,
                    "lambda_end": self.model_config.lambda_end,
                    "use_time_dependent_consistency": self.model_config.use_time_dependent_consistency,
                    "lambda_cons": self.model_config.lambda_cons,
                    "continuous_final_time": self.model_config.continuous_final_time,
                    "adversarial": {
                        "enabled": self.model_config.adversarial.enabled,
                        "lambda_adv": self.model_config.adversarial.lambda_adv,
                        "hidden_dims": self.model_config.adversarial.hidden_dims,
                        "activation": self.model_config.adversarial.activation,
                        "batch_norm": self.model_config.adversarial.batch_norm,
                        "dropout": self.model_config.adversarial.dropout,
                    },
                    "state_network": _network_config_to_dict(self.model_config.state_network),
                    "context_network": _network_config_to_dict(self.model_config.context_network),
                    "decoder_network": _network_config_to_dict(self.model_config.decoder_network),
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: torch.device) -> "TensorProductKoopmanFlow":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = TensorProductKoopmanModelConfig(
            latent_dim=model_config_dict.get("latent_dim", 256),
            context_feature_dim=model_config_dict.get("context_feature_dim", 128),
            tensor_rank=model_config_dict.get("tensor_rank", 64),
            lambda_ae=model_config_dict.get("lambda_ae", 1.0),
            lambda_lat=model_config_dict.get("lambda_lat", 1.0),
            lambda_end=model_config_dict.get("lambda_end", 1.0),
            use_time_dependent_consistency=model_config_dict.get(
                "use_time_dependent_consistency",
                model_config_dict.get("use_full_trajectory", False),
            ),
            lambda_cons=model_config_dict.get("lambda_cons", 1.0),
            continuous_final_time=model_config_dict.get("continuous_final_time", 1.0),
            adversarial=AdversarialConfig(**model_config_dict.get("adversarial", {})),
            state_network=NetworkConfig(**model_config_dict["state_network"]),
            context_network=NetworkConfig(**model_config_dict["context_network"]),
            decoder_network=NetworkConfig(**model_config_dict["decoder_network"]),
        )
        model = cls(
            input_dim=checkpoint["input_dim"],
            context_dim=checkpoint["context_dim"],
            model_config=model_config,
            device=device,
        )
        state_dict = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if not key.startswith("teacher_model.")
        }
        if any(key.startswith("generator_") for key in state_dict):
            raise ValueError(
                f"{filepath} was trained with the old diagonal continuous-time generator, which has been replaced "
                "by the tensor-product generator; retrain the tensor-product Koopman model."
            )
        model.load_state_dict(state_dict)
        model.to(device)
        return model
