import argparse
import yaml
import os
from os.path import join
import torch
from torch.utils.data import DataLoader

from sbi_dataset import generate_dataset, load_dataset
from utils import evaluate_model
from conditional_flow_matching import ConditionalFlowMatching


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching model for SBI")
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--load_existing", action="store_true", default=False,
        help="Load existing dataset if available"
    )
    parser.add_argument(
        "--compute_c2st", action="store_true", default=False,
        help="Compute C2ST scores during evaluation (slow, default: False)"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set up dataset-specific paths
    dataset_name = config["task"]["name"]
    base_logs_dir = "logs"
    dataset_dir = join(base_logs_dir, dataset_name, "data")
    
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Training Conditional Flow Matching model for task: {dataset_name}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Generate or load dataset
    if args.load_existing and os.path.exists(join(dataset_dir, "theta.npy")):
        print("Loading existing dataset...")
        train_dataset, val_dataset = load_dataset(
            dataset_dir, 
            config, 
            train_fraction=config["task"]["train_fraction"]
        )
    else:
        print("Generating new dataset...")
        train_dataset, val_dataset = generate_dataset(
            config,
            batch_size=config["task"]["batch_size"],
            directory_save=dataset_dir,
            train_fraction=config["task"]["train_fraction"]
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    # Train model
    print("Starting training...")
    dataset_name = config["task"]["name"]
    model_dir = join("logs", dataset_name, "flow_matching", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create ConditionalFlowMatching model
    model = ConditionalFlowMatching(
        input_dim=config["task"]["dim_theta"],
        context_dim=config["task"]["dim_x"],
        posterior_kwargs=config["model"]["posterior_kwargs"],
        device=config["training"]["device"],
        output_dir=join("logs", dataset_name, "flow_matching")
    )

    # Set optimizer and scheduler
    model.optimizer_kwargs = config["training"]["optimizer"]
    model.scheduler_kwargs = config["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()
    
    # Train model
    model.train_model(
        train_loader,
        val_loader,
        model_dir,  # train_dir as positional argument
        config["training"]["epochs"],  # epochs as positional argument
        early_stopping=config["training"]["early_stopping"],
        use_tensorboard=config["training"]["use_tensorboard"],
        patience=config["training"]["patience"]
    )

    # Load the best model
    best_model = ConditionalFlowMatching.load(
        join(model_dir, "best_model.pt"),
        device=config["training"]["device"]
    )
    
    print("Training completed!")
    
    # Evaluate model
    if args.compute_c2st:
        print("Running evaluation with C2ST computation...")
        c2st_scores, flow_times = evaluate_model(
            config, train_dataset, best_model, compute_c2st=True, model_type="flow_matching"
        )
        print("Evaluation completed!")
    else:
        print("Running quick evaluation (skipping C2ST)...")
        c2st_scores, flow_times = evaluate_model(
            config, train_dataset, best_model, compute_c2st=False, model_type="flow_matching"
        )
        print("Evaluation completed!")


if __name__ == "__main__":
    main()