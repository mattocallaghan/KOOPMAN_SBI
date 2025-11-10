import csv
import math
import time
from os.path import join
import os

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st, ksd, mmd, posterior_mean_error, posterior_variance_ratio, median_distance
import torch
import numpy as np
import yaml

from koopman_flow import KoopmanFlow





def evaluate_model(config, dataset, model, compute_c2st=False, model_type="koopman"):
    """Evaluate model using YAML configuration."""
    
    # Create evaluation directory based on dataset name and model type
    dataset_name = config["task"]["name"]
    eval_dir = join("logs", dataset_name, model_type, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    task = sbibm.get_task(config["task"]["name"])

    c2st_scores = {}
    flow_times = []  # Track sampling times
    
    print("Generating posterior samples and plots...")
    for obs in range(1, 11):
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = 10000  # Generate 1000 teacher samples for evaluation
        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        # Time flow sampling
        start_time = time.time()
        
        # generate (num_samples * 2), to account for samples outside of the prior
        posterior_samples = model.sample_batch(observation.repeat((num_samples * 2, 1)))
        
        flow_time = time.time() - start_time
        flow_times.append(flow_time)
        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )

        # discard samples outside the prior
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        print(
            f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% of the samples "
            f"lie outside of the prior. Discarding these."
        )
        posterior_samples = posterior_samples[prior_mask]

        n = min(len(reference_samples), len(posterior_samples))
        
        # Only compute C2ST if requested
        if compute_c2st:
            c2st_score = c2st(posterior_samples[:n], reference_samples[:n])
            c2st_scores[f"C2ST {obs}"] = c2st_score.item()
            print(f"C2ST score for observation {obs}: {c2st_score.item():.3f}")
            title = f"Observation {obs}: C2ST = {c2st_score.item():.3f}, Time = {flow_time:.3f}s ({num_samples*2} generated)"
        else:
            title = f"Observation {obs}: Time = {flow_time:.3f}s ({num_samples*2} generated)"
        
        # Always generate plots
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            posterior_samples[:, 0],
            posterior_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label=f"Koopman flow ({len(posterior_samples)} samples)",
        )
        plt.scatter(
            reference_samples[:, 0],
            reference_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label=f"reference ({len(reference_samples)} samples)",
        )
        plt.title(title)
        plt.legend()
        plt.savefig(join(eval_dir, f"posterior_{obs}.png"))
        plt.close()  # Close figure to prevent memory buildup
    
    if not compute_c2st:
        print("Skipped C2ST computation")
    
    # Speed summary and plot for flow model
    avg_time = np.mean(flow_times)
    print(f"Average Koopman flow sampling time: {avg_time:.3f} ± {np.std(flow_times):.3f} seconds")
    
    # Speed comparison plot with sample information
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), flow_times, 'ro-', label='Koopman Flow Sampling Time')
    
    # Add sample count annotations
    for i, (obs, time_val) in enumerate(zip(range(1, 11), flow_times)):
        # We generate num_samples * 2 = 1000 * 2 = 2000 samples
        sample_count = 10000 * 2  # Actual number of samples generated
        ax.annotate(f'{sample_count}', (obs, time_val), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Observation Number')
    ax.set_ylabel('Sampling Time (seconds)')
    ax.set_title('Koopman Flow Sampling Speed (numbers show sample count generated)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(join(eval_dir, "sampling_speed.png"), dpi=150, bbox_inches='tight')
    plt.close()

    with open(
        join(eval_dir, "c2st.csv"), "w"
    ) as f:
        w = csv.DictWriter(f, c2st_scores.keys())
        w.writeheader()
        w.writerow(c2st_scores)

    return c2st_scores, flow_times



def plot_posteriors_and_log_probs(
        reference_samples,
        posterior_samples,
        reference_log_probs,
        posterior_log_probs,
        train_dir
        ):
    plt.hist(
        posterior_log_probs,
        alpha=0.2,
        label="posterior log probs",
    )
    plt.hist(
        reference_log_probs,
        alpha=0.2,
        label="reference log probs",
    )
    plt.legend()
    plt.savefig(join(train_dir, "log_probs.png"))
    plt.clf()

    plt.scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=0.5,
        alpha=0.2,
        label="flow matching",
    )
    plt.scatter(
        reference_samples[:, 0],
        reference_samples[:, 1],
        s=0.5,
        alpha=0.2,
        label="reference",
    )
    plt.legend()
    plt.savefig(join(train_dir, "posteriors.png"))


def complete_model_evaluation(config, dataset, model, metrics, save_samples=True, model_type="koopman"):
    """Complete evaluation of model using YAML configuration."""
    
    # Create evaluation directory based on dataset name and model type
    dataset_name = config["task"]["name"]
    eval_dir = join("logs", dataset_name, model_type, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    task = sbibm.get_task(config["task"]["name"])
    max_batch_size = config["task"].get("max_batch_size", 500)
    metrics_dict = {'c2st': c2st, 'ksd': ksd, 'mmd': mmd, 'posterior_mean_error': posterior_mean_error,
                     'posterior_variance_ratio': posterior_variance_ratio, 'median_distance': median_distance}
    metrics = [m for m in metrics if m in metrics_dict.keys()]
    result_list = []

    for obs in range(1, 10):
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = len(reference_samples)
        reference_samples_standardized = dataset.standardize(
            reference_samples, label="theta"
        )

        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        reference_log_probs = []
        for i in range(math.ceil(num_samples / max_batch_size)):
            reference_batch = reference_samples_standardized[(i*max_batch_size):((i+1)*max_batch_size)]
            # We evaluate likelihoods of the standardized data
            reference_log_probs.append(model.log_prob_batch(
                    reference_batch, observation.repeat((len(reference_batch), 1))
            ).detach())
        reference_log_probs = torch.cat(reference_log_probs, dim=0)
        # generate (num_samples * 2), to account for samples outside of the prior
        posterior_samples, posterior_log_probs = [], []

        for i in range(2 * num_samples // max_batch_size + 1):
            posterior_samples_batch, posterior_log_probs_batch = model.sample_and_log_prob_batch(
                observation.repeat((max_batch_size, 1))
            )
            posterior_samples.append(posterior_samples_batch.detach())
            posterior_log_probs.append(posterior_log_probs_batch.detach())
        posterior_samples = torch.cat(posterior_samples, dim=0)
        posterior_log_probs = torch.cat(posterior_log_probs, dim=0)

        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )

        # discard samples outside the prior
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        print(
            f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% of the samples "
            f"lie outside of the prior. Discarding these."
        )
        posterior_samples = posterior_samples[prior_mask]
        posterior_log_probs = posterior_log_probs[prior_mask]
        n = min(len(reference_samples), len(posterior_samples))
        if len(reference_samples) > len(posterior_samples):
            print('Less posterior samples than reference samples!')
        posterior_samples = posterior_samples[:n].detach()
        posterior_log_probs = posterior_log_probs[:n].detach()
        reference_samples = reference_samples[:n].detach()
        reference_log_probs = reference_log_probs[:n].detach()

        if obs == 1:
            plot_posteriors_and_log_probs(reference_samples, posterior_samples, reference_log_probs,
                                          posterior_log_probs, eval_dir)
        result = {'num_observation': obs}
        for m in metrics:
            if m == 'ksd':
                score = ksd(task, obs, posterior_samples)
            else:
                score = metrics_dict[m](posterior_samples, reference_samples).item()
            result[m] = score
        result_list.append(result)

        if save_samples:
            dir_obs = join(eval_dir, str(obs).zfill(2))
            from pathlib import Path
            Path(dir_obs).mkdir(exist_ok=True)
            np.save(join(dir_obs, 'samples.npy'), posterior_samples)
            np.save(join(dir_obs, 'posterior_log_probs.npy'), posterior_log_probs)
            np.save(join(dir_obs, 'reference_log_probs.npy'), reference_log_probs)

    with open(
            join(eval_dir, "results.csv"), "w"
    ) as f:
        w = csv.DictWriter(f, result_list[0].keys())
        w.writeheader()
        w.writerows(result_list)
    
    return result_list


def compute_validation_loss(config, model, validation_loader, model_type="koopman"):
    """Compute validation loss for model using YAML configuration."""
    
    # Create evaluation directory based on dataset name and model type
    dataset_name = config["task"]["name"]
    eval_dir = join("logs", dataset_name, model_type, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    val_loss = model.validation_epoch(validation_loader)
    
    # Note: Koopman model doesn't implement log_prob_batch properly yet
    # This is a simplified version that just returns the validation loss
    validation_losses = [{'val_loss': val_loss}]
    
    import pandas as pd
    df = pd.DataFrame.from_records(validation_losses)
    df.to_csv(join(eval_dir, 'val_losses.csv'), index=False)
    
    return val_loss