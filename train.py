import os    
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='x'

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.gpt_neo import GPTNeoWithSelfAblation
from model.config import GPTNeoWithSelfAblationConfig, TrainingConfig, WandBConfig
from utils.data_preparation import prepare_data
from utils.training import BatchGenerator, LossEstimator
from utils.parser import return_parser

import numpy as np
import wandb

from dotenv import load_dotenv

def train_gptneo(model, config, model_name=None):

    # PLEASE update these values for each real training run you do - it will really help us keep track
    extra_config = {
        "dataset_name": "TinyStories",
        "ablation_processing": "soft-top-K-version-1",
        "top_k_level": "layer-by-layer",
        "per_layer_ablation_position": "pre"
    }
    wandb_config = {**vars(model.config), **vars(config), **extra_config}

    wandb.init(project="gpt-neo-self-ablation", config=wandb_config, name=model_name)

    train_batch_gen = BatchGenerator(config.train_file, config.block_size, config.batch_size, config.device)
    val_batch_gen = BatchGenerator(config.val_file, config.block_size, config.batch_size, config.device)
    loss_estimator = LossEstimator(model, train_batch_gen, val_batch_gen, config.eval_iters, config.device)
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_batches)
    best_val_loss = float('inf')
    
    for iteration in tqdm(range(config.num_batches)):
        model.train()
        # Get batch
        x, y = train_batch_gen.get_batch()
        
        # Forward pass
        train_outputs = model(x, targets=y)
        loss = train_outputs['loss']
        print('k: ', train_outputs['effective_k'])
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config.max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # log progress every log_interval iterations
        if (iteration + 1) % config.log_interval == 0:
            stats = loss_estimator.estimate_loss()
            print(f"Iteration {iteration}: train loss {stats['train']['loss']:.4f}, val loss {stats['val']['loss']:.4f}")
            wandb.log(stats | {"iteration": iteration, "current_learning_rate": optimizer.param_groups[0]['lr']})

            # Save best model
            if stats['val']['loss'] < best_val_loss:
                best_val_loss = stats['val']['loss']
                torch.save(model.state_dict(), config.save_path)
                print(f"New best model saved to {config.save_path}")
                wandb.save(config.save_path) # Save the model to wandb
                print(f"Model saved to wandb")

    print("Training completed!")
    wandb.finish() # Finish the wandb run

if __name__ == "__main__":

    print("Loading environment variables")
    load_dotenv()
    
    # Gets arguments from command line
    parser = return_parser()
    args = parser.parse_args()
    
    model_name = args.model_name
    print(model_name)
    
    # Creates a dictionary with the arguments except model_name
    args = vars(args)
    del args['model_name']
    
    # Set up configuration
    model_config = GPTNeoWithSelfAblationConfig(hidden_size=128, **args)
        
    training_config = TrainingConfig(batch_size=32, save_path=f"model_weights/{model_name}.pt")

    # Initialize model
    model = GPTNeoWithSelfAblation(model_config)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # Prepare data
    print("Preparing data")
    prepare_data(output_file=training_config.train_file)
    print("Okay")
    prepare_data(split='validation', output_file=training_config.val_file)

    # Train model
    print("Beginning training")
    train_gptneo(model, training_config, model_name)
