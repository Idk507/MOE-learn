#!/usr/bin/env python3
"""
Main script demonstrating how to use the MoE Transformer implementation.
"""

import torch
from config import Config
from data_utils import DataLoader
from model import TinyGPT
from trainer import Trainer
from utils import set_seed, print_model_info, print_device_info

def main():
    """Main function to train and evaluate the MoE Transformer."""
    
    # Initialize configuration
    config = Config()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Print device information
    print_device_info()
    
    # Load and prepare data
    print("Loading data...")
    data_loader = DataLoader(config)
    vocab_info = data_loader.get_vocab_info()
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Training data size: {len(data_loader.train_data):,} tokens")
    print(f"Validation data size: {len(data_loader.val_data):,} tokens")
    
    # Initialize model
    print("\nInitializing model...")
    model = TinyGPT(config, vocab_info['vocab_size'])
    
    # Print model information
    print_model_info(model, config)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, config, data_loader)
    
    # Train the model
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    training_metrics = trainer.train(
        eval_interval=100,  # Evaluate every 100 steps
        log_interval=50     # Log every 50 steps
    )
    
    # Generate some text
    print("\n" + "="*50)
    print("TEXT GENERATION")
    print("="*50)
    
    prompts = [
        "JULIET: ",
        "ROMEO: ",
        "To be or not to be, ",
        "Once upon a time, "
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        generated_text = trainer.generate_text(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.8,
            top_k=50
        )
        
        # Print only the generated part (exclude the original prompt)
        generated_only = generated_text[len(prompt):]
        print(f"Generated: {generated_only}")
    
    # Print final training statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Final training loss: {training_metrics['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {training_metrics['val_losses'][-1]:.4f}")
    print(f"Final auxiliary loss: {training_metrics['aux_losses'][-1]:.4f}")
    
    # Optionally save the model
    save_model = input("\nSave model? (y/n): ").lower().strip() == 'y'
    if save_model:
        from utils import save_model as save_model_fn, save_config
        
        model_path = "checkpoints/model.pt"
        config_path = "checkpoints/config.json"
        
        save_model_fn(model, config, model_path)
        save_config(config, config_path)
        print("Model and configuration saved!")

def compare_moe_vs_dense():
    """Compare MoE model against dense baseline."""
    print("\n" + "="*60)
    print("COMPARING MoE vs DENSE MODELS")
    print("="*60)
    
    results = {}
    
    for use_moe in [False, True]:
        model_type = "MoE" if use_moe else "Dense"
        print(f"\nTraining {model_type} model...")
        
        # Create config
        config = Config()
        config.use_moe = use_moe
        config.max_steps = 200  # Shorter training for comparison
        set_seed(config.seed)
        
        # Load data
        data_loader = DataLoader(config)
        vocab_size = data_loader.get_vocab_info()['vocab_size']
        
        # Initialize model and trainer
        model = TinyGPT(config, vocab_size)
        trainer = Trainer(model, config, data_loader)
        
        print(f"{model_type} - Total parameters: {model.count_parameters()/1e6:.2f}M")
        
        # Train model
        metrics = trainer.train(eval_interval=50, log_interval=25)
        
        # Store results
        results[model_type] = {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'total_params': model.count_parameters(),
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for model_type, metrics in results.items():
        print(f"\n{model_type} Model:")
        print(f"  Parameters: {metrics['total_params']/1e6:.2f}M")
        print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {metrics['final_val_loss']:.4f}")
    
    # Performance comparison
    if 'Dense' in results and 'MoE' in results:
        dense_val_loss = results['Dense']['final_val_loss']
        moe_val_loss = results['MoE']['final_val_loss']
        
        improvement = ((dense_val_loss - moe_val_loss) / dense_val_loss) * 100
        
        print(f"\nValidation Loss Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print("✓ MoE model performs better!")
        else:
            print("✗ Dense model performs better (may need more training steps)")

if __name__ == "__main__":
    # Run main training
    main()
    
    # Optionally run comparison
    run_comparison = input("\nRun MoE vs Dense comparison? (y/n): ").lower().strip() == 'y'
    if run_comparison:
        compare_moe_vs_dense()
