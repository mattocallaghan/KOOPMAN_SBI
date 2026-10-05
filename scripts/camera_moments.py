"""Compare the moments of the camera student's posteriors with the teacher's.

Run from the repo root (any machine; picks CUDA, MPS or CPU):
    python scripts/camera_moments.py [--student PATH] [--components 100]

Per observation, the teacher's 1000 cached samples are split into thirds: one third fits the teacher's
principal directions, one third is the reference, and the last third gives the teacher-vs-teacher noise
floor. The student is compared with the same reference in the teacher's top principal components:
mean shift, log variance ratio, skewness, kurtosis and Frechet distance, averaged over the observations.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from koopman_sbi.config import load_experiment_config
from koopman_sbi.data import load_or_generate_dataset
from koopman_sbi.image_evaluation import teacher_posterior_samples
from koopman_sbi.models.tensorproduct_koopman import TensorProductKoopmanFlow
from koopman_sbi.runtime import detect_device
from koopman_sbi.tasks import get_task


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frechet (Gaussian W2^2) distance between Gaussian fits of two (n, K) sample sets."""
    ma, mb, ca, cb = a.mean(0), b.mean(0), torch.cov(a.T), torch.cov(b.T)
    vals, vecs = torch.linalg.eigh(ca)
    root = vecs @ torch.diag(vals.clamp_min(0).sqrt()) @ vecs.T
    cross = torch.linalg.eigvalsh(root @ cb @ root).clamp_min(0).sqrt().sum()
    return float((ma - mb).pow(2).sum() + ca.trace() + cb.trace() - 2 * cross)


def moments(samples: torch.Tensor, reference: torch.Tensor) -> dict:
    standardize = lambda x: (x - x.mean(0)) / x.std(0)
    ratio = samples.var(0) / reference.var(0)
    return {
        "mean shift (ref std units)": float(((samples.mean(0) - reference.mean(0)) / reference.std(0)).abs().mean()),
        "|log variance ratio|": float(ratio.log().abs().mean()),
        "log variance ratio (signed)": float(ratio.log().mean()),
        "|skewness diff|": float(((standardize(samples) ** 3).mean(0) - (standardize(reference) ** 3).mean(0)).abs().mean()),
        "|kurtosis diff|": float(((standardize(samples) ** 4).mean(0) - (standardize(reference) ** 4).mean(0)).abs().mean()),
        "Frechet / ref total variance": frechet(samples, reference) / float(reference.var(0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="koopman_sbi/configs/tasks/camera_model.yaml")
    parser.add_argument("--student", default="logs/camera_model/last_model/train_tensorproduct_koopman/best_model.pt")
    parser.add_argument("--components", type=int, default=100)
    args = parser.parse_args()

    device = detect_device("auto")
    torch.manual_seed(0)
    config = load_experiment_config(args.config)
    bundle = load_or_generate_dataset(config)
    standardizer, task = bundle.standardizer, get_task(config.task.name)
    student = TensorProductKoopmanFlow.load(args.student, device=device)
    print(f"student latent_dim {student.model_config.latent_dim}, device {device}", flush=True)

    rows = {"student": [], "teacher": []}
    for obs in config.evaluation.observations:
        teacher = teacher_posterior_samples(config, bundle, obs, 1000, device).double()
        context = standardizer.standardize_x(task.get_observation(obs)).repeat(1000, 1).to(device)
        with torch.no_grad():
            samples = standardizer.inverse_theta(student.sample_batch(context).cpu()).double().clamp(0, 1)
        order = torch.randperm(1000)
        fit, reference, baseline = teacher[order[:334]], teacher[order[334:667]], teacher[order[667:]]
        centre = fit.mean(0)
        _, vectors = torch.linalg.eigh(torch.cov((fit - centre).T))
        top = vectors.flip(1)[:, : args.components]
        project = lambda x: (x - centre) @ top
        rows["student"].append(moments(project(samples[:333]), project(reference)))
        rows["teacher"].append(moments(project(baseline), project(reference)))
        print(f"observation {obs} done", flush=True)

    print(f"\ntop {args.components} teacher principal components, averaged over "
          f"{len(config.evaluation.observations)} observations (333 samples per set)")
    print(f"{'':32s} {'student':>10s} {'teacher vs teacher (noise floor)':>34s}")
    for key in rows["student"][0]:
        student_value = np.mean([row[key] for row in rows["student"]])
        floor_value = np.mean([row[key] for row in rows["teacher"]])
        print(f"{key:32s} {student_value:10.3f} {floor_value:34.3f}")


if __name__ == "__main__":
    main()
