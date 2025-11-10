import argparse
import yaml
import os
from os.path import join
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sbi_dataset import generate_dataset, load_dataset
from conditional_flow_matching import ConditionalFlowMatching
from koopman_flow import KoopmanFlow
from utils import evaluate_model


class KoopmanTeacherDataset(Dataset):
    """Dataset for Koopman training using teacher-generated trajectories.
     Assumes that the data sampled from the teacher model is standardized"""
    
    def __init__(self, eps, theta, x):
        super(KoopmanTeacherDataset, self).__init__()
        self.eps = eps
        self.theta = theta
        self.x = x
    
    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.eps[idx], self.theta[idx], self.x[idx]


def generate_teacher_trajectories(teacher_model, context_data, pipeline_config, device):
    """Generate (eps, theta, x) training data using the teacher flow matching model.
    
    For each of num_samples trajectories:
    1. Sample random noise eps ~ N(0, I) 
    2. Sample random context x from context_data
    3. Evolve eps through the flow to get theta using sample_batch(context, custom_theta_0=eps)
    """
    teacher_model.eval()
    
    num_samples = pipeline_config["teacher"]["num_samples"]
    batch_size = pipeline_config["teacher"]["batch_size"]
    
    eps_list = []
    theta_list = []
    x_list = []
    
    print(f"Generating {num_samples} teacher trajectories...")
    
    with torch.no_grad():
        # Generate data in batches to avoid memory issues
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample random context from the provided context data
            context_indices = torch.randint(0, len(context_data), (current_batch_size,))
            context_batch = context_data[context_indices].to(device)
            
            # Sample random noise eps for each trajectory
            eps_batch = teacher_model.sample_theta_0(current_batch_size)  # theta_0 ~ N(0, I)
            
            # Evolve eps through the flow to get theta using custom_theta_0
            theta_batch = teacher_model.sample_batch(context_batch, custom_theta_0=eps_batch)
            
            eps_list.append(eps_batch.cpu())   # Starting noise
            theta_list.append(theta_batch.cpu())  # Final evolved theta
            x_list.append(context_batch.cpu())    # Context
            
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Generated {min((i + 1) * batch_size, num_samples):,} / {num_samples:,} samples")
    
    eps_tensor = torch.cat(eps_list, dim=0)[:num_samples]
    theta_tensor = torch.cat(theta_list, dim=0)[:num_samples]
    x_tensor = torch.cat(x_list, dim=0)[:num_samples]
    
    return eps_tensor, theta_tensor, x_tensor


def load_saved_trajectories(teacher_data_dir):
    """Load previously saved teacher trajectories."""
    print(f"Loading saved trajectories from: {teacher_data_dir}")
    
    eps_path = join(teacher_data_dir, 'eps.npy')
    theta_path = join(teacher_data_dir, 'theta.npy')
    x_path = join(teacher_data_dir, 'x.npy')
    
    if not all(os.path.exists(path) for path in [eps_path, theta_path, x_path]):
        raise FileNotFoundError(f"Saved trajectory files not found in {teacher_data_dir}")
    
    eps_tensor = torch.tensor(np.load(eps_path), dtype=torch.float)
    theta_tensor = torch.tensor(np.load(theta_path), dtype=torch.float)
    x_tensor = torch.tensor(np.load(x_path), dtype=torch.float)
    
    print(f"Loaded saved trajectories:")
    print(f"  eps shape: {eps_tensor.shape}")
    print(f"  theta shape: {theta_tensor.shape}")
    print(f"  x shape: {x_tensor.shape}")
    
    return eps_tensor, theta_tensor, x_tensor


def main():
    parser = argparse.ArgumentParser(description="Pipeline for training Koopman model with teacher data")
    parser.add_argument(
        "--koopman_config", required=True, help="Path to Koopman configuration file"
    )
    parser.add_argument(
        "--flow_config", required=True, help="Path to Flow Matching configuration file"
    )
    parser.add_argument(
        "--pipeline_config", required=True, help="Path to Pipeline configuration file"
    )

    args = parser.parse_args()

    # Load configurations
    with open(args.koopman_config, "r") as f:
        koopman_config = yaml.safe_load(f)
    
    with open(args.flow_config, "r") as f:
        flow_config = yaml.safe_load(f)
        
    with open(args.pipeline_config, "r") as f:
        pipeline_config = yaml.safe_load(f)
    
    
    # Set up paths
    dataset_name = koopman_config["task"]["name"]
    base_logs_dir = "logs"
    dataset_dir = join(base_logs_dir, dataset_name, "data")
    
    # Update pipeline config paths
    pipeline_config["paths"]["flow_model_path"] = join(base_logs_dir, dataset_name, "flow_matching", "models", "best_model.pt")
    pipeline_config["paths"]["koopman_model_dir"] = join(base_logs_dir, dataset_name, "koopman", "models")
    pipeline_config["paths"]["base_data_dir"] = dataset_dir
    pipeline_config["data"]["teacher_data_dir"] = join(base_logs_dir, dataset_name, "teacher_data")
    
    print(f"Pipeline for task: {dataset_name}")
    
    # Check if this is evaluation-only mode
    evaluation_only = pipeline_config["training"].get("evaluation_only", False)
    
    if evaluation_only:
        print("Running in EVALUATION-ONLY mode - skipping trajectory generation and training")
    else:
        # Load the trained flow matching model
        # this assumes that the teacher model was already trained and saved
        flow_model_path = pipeline_config["paths"]["flow_model_path"]
        if not os.path.exists(flow_model_path):
            raise FileNotFoundError(f"Flow matching model not found at {flow_model_path}. Train it first using train_flow_matching.py!")
        
        print(f"Loading flow matching teacher model from: {flow_model_path}")
        teacher_model = ConditionalFlowMatching.load(
            flow_model_path, 
            device=flow_config["training"]["device"]
        )
    
    # Load the original dataset for evaluation and context (always needed)
    if os.path.exists(join(dataset_dir, "theta.npy")):
        print("Loading existing dataset...")
        train_dataset, val_dataset = load_dataset(
            dataset_dir, 
            flow_config, 
            train_fraction=flow_config["task"]["train_fraction"]
        )
    else:
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Generate it first using train_flow_matching.py!")

    if not evaluation_only:
        # Extract context data (x) from the dataset for trajectory generation
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # For flow matching dataset: (theta, x) pairs
        train_context = train_batch[1]  # x values
        val_context = val_batch[1]      # x values
        all_context = torch.cat([train_context, val_context], dim=0)
        
        # Randomly select num_context points from all available context
        num_context = pipeline_config["teacher"]["num_context"]
        if num_context > len(all_context):
            print(f"Warning: num_context ({num_context}) > available context ({len(all_context)}). Using all available.")
            selected_context = all_context
        else:
            context_indices = torch.randperm(len(all_context))[:num_context]
            selected_context = all_context[context_indices]
        
        print(f"Selected {len(selected_context)} context points from {len(all_context)} available")
        
        # Check if we should load existing trajectories or generate new ones
        teacher_data_dir = pipeline_config["data"]["teacher_data_dir"]
        
        if pipeline_config["data"]["load_existing_trajectories"]:
            # Load existing saved trajectories
            try:
                eps_teacher, theta_teacher, x_teacher = load_saved_trajectories(teacher_data_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Falling back to generating new trajectories...")
                eps_teacher, theta_teacher, x_teacher = generate_teacher_trajectories(
                    teacher_model, 
                    selected_context, 
                    pipeline_config,
                    flow_config["training"]["device"]
                )
        else:
            # Generate new teacher trajectories: (eps=theta_0, theta=theta_1, x=context)
            eps_teacher, theta_teacher, x_teacher = generate_teacher_trajectories(
                teacher_model, 
                selected_context, 
                pipeline_config,
                flow_config["training"]["device"]
            )
            
        print(f"Generated teacher trajectories:")
        print(f"  eps (theta_0) shape: {eps_teacher.shape}")
        print(f"  theta (theta_1) shape: {theta_teacher.shape}")
        print(f"  x (context) shape: {x_teacher.shape}")
        
        # Save teacher data if requested
        if pipeline_config["data"]["save_teacher_data"]:
            os.makedirs(teacher_data_dir, exist_ok=True)
            
            np.save(join(teacher_data_dir, 'eps.npy'), eps_teacher.numpy())
            np.save(join(teacher_data_dir, 'theta.npy'), theta_teacher.numpy())
            np.save(join(teacher_data_dir, 'x.npy'), x_teacher.numpy())
            print(f"Saved teacher data to: {teacher_data_dir}")
        
        # Split teacher data into train/val
        num_samples = len(eps_teacher)
        num_train = int(num_samples * pipeline_config["data"]["train_fraction"])
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        # Create subset datasets
        train_eps = eps_teacher[train_indices]
        train_theta = theta_teacher[train_indices]
        train_x = x_teacher[train_indices]
        
        val_eps = eps_teacher[val_indices]
        val_theta = theta_teacher[val_indices] 
        val_x = x_teacher[val_indices]
        
        koopman_train_dataset = KoopmanTeacherDataset(train_eps, train_theta, train_x)
        koopman_val_dataset = KoopmanTeacherDataset(val_eps, val_theta, val_x)
        
        print(f"Created Koopman datasets - Train: {len(koopman_train_dataset)}, Val: {len(koopman_val_dataset)}")
    
        # Train Koopman model if requested
        if pipeline_config["training"]["run_koopman_training"]:
            print("\\n" + "="*50)
            print("Starting Koopman model training...")
            print("="*50)
            
            # Create data loaders for Koopman training
            koopman_train_loader = DataLoader(
                koopman_train_dataset,
                batch_size=koopman_config["training"]["batch_size"],
                shuffle=True,
                num_workers=koopman_config["training"]["num_workers"]
            )
            
            koopman_val_loader = DataLoader(
                koopman_val_dataset,
                batch_size=koopman_config["training"]["batch_size"],
                shuffle=False,
                num_workers=koopman_config["training"]["num_workers"]
            )
            
            # Set dimensions in config
            koopman_config["task"]["dim_theta"] = koopman_train_dataset.theta.shape[1]
            koopman_config["task"]["dim_x"] = koopman_train_dataset.x.shape[1]
            
            # Create model directory
            model_dir = pipeline_config["paths"]["koopman_model_dir"]
            os.makedirs(model_dir, exist_ok=True)
            
            # Create Koopman model
            koopman_model = KoopmanFlow(
                input_dim=koopman_config["task"]["dim_theta"],
                context_dim=koopman_config["task"]["dim_x"],
                lifting_dim=koopman_config["model"]["lifting_dim"],
                network_kwargs=koopman_config["model"]["network_kwargs"],
                device=koopman_config["training"]["device"],
                lambda_rec=koopman_config["model"]["lambda_rec"],
                lambda_lat=koopman_config["model"]["lambda_lat"],
                lambda_pred=koopman_config["model"]["lambda_pred"],
                output_dir=join("logs", dataset_name, "koopman")
            )

            # Set optimizer and scheduler
            koopman_model.optimizer_kwargs = koopman_config["training"]["optimizer"]
            koopman_model.scheduler_kwargs = koopman_config["training"]["scheduler"]
            koopman_model.initialize_optimizer_and_scheduler()
            
            # Train model
            koopman_model.train_model(
                koopman_train_loader,
                koopman_val_loader,
                model_dir,
                koopman_config["training"]["epochs"],
                early_stopping=koopman_config["training"]["early_stopping"],
                use_tensorboard=koopman_config["training"]["use_tensorboard"],
                patience=koopman_config["training"]["patience"]
            )

            print("Koopman training completed!")
    
    # Evaluation section (runs in both modes)
    if pipeline_config["training"]["run_evaluation"]:
        print("\\n" + "="*50)
        print("Starting model evaluation...")
        print("="*50)
        
        # Load the best model
        model_dir = pipeline_config["paths"]["koopman_model_dir"]
        best_koopman_model = KoopmanFlow.load(
            join(model_dir, "best_model.pt"),
            device=koopman_config["training"]["device"]
        )
        
        compute_c2st = pipeline_config["training"]["compute_c2st"]
        if compute_c2st:
            print("Running evaluation with C2ST computation...")
        else:
            print("Running quick evaluation (skipping C2ST)...")
        
        # Use the original SBI dataset for evaluation, not the teacher dataset
        c2st_scores, flow_times = evaluate_model(
            koopman_config, train_dataset, best_koopman_model, 
            compute_c2st=compute_c2st, model_type="koopman"
        )
        print("Evaluation completed!")
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()