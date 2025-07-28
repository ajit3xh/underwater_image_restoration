import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import pandas as pd

from model import CNN_ViT_UnderwaterRestorer
from dataset import myDataset  # custom dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


# === Color Constancy Loss Function ===
def color_constancy_loss(img):
    mean_rgb = torch.mean(img, dim=[2, 3], keepdim=True)
    mean_diff = torch.abs(mean_rgb - torch.mean(mean_rgb, dim=1, keepdim=True))
    return torch.mean(mean_diff)


# === Calculate PSNR for accuracy metric ===
def calculate_psnr(output, target):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# === Validation Function ===
def validate(model, val_loader, device, l1_loss_fn, ssim_loss_fn):
    """Validate the model and return average loss and accuracy metrics"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            
            # Calculate losses
            l1 = l1_loss_fn(outputs, targets)
            ssim_val = ssim_loss_fn(outputs, targets)
            ssim_loss = 1 - ssim_val
            color_loss = color_constancy_loss(outputs)
            loss = l1 + 0.2 * ssim_loss + 0.1 * color_loss
            
            # Calculate accuracy metrics
            psnr = calculate_psnr(outputs, targets)
            
            total_loss += loss.item()
            total_psnr += psnr.item() if psnr != float('inf') else 40.0  # Cap PSNR at 40 for averaging
            total_ssim += ssim_val.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    return avg_loss, avg_psnr, avg_ssim


# === Plot and Save Training Graphs ===
def save_training_plots(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim, save_dir):
    """Create and save training visualization plots"""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss Plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR Plot (Accuracy Metric)
    axes[0, 1].plot(epochs, train_psnr, 'g-', label='Training PSNR', linewidth=2)
    axes[0, 1].plot(epochs, val_psnr, 'orange', label='Validation PSNR', linewidth=2)
    axes[0, 1].set_title('PSNR (Peak Signal-to-Noise Ratio)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM Plot
    axes[1, 0].plot(epochs, train_ssim, 'purple', label='Training SSIM', linewidth=2)
    axes[1, 0].plot(epochs, val_ssim, 'brown', label='Validation SSIM', linewidth=2)
    axes[1, 0].set_title('SSIM (Structural Similarity Index)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined Accuracy Plot
    axes[1, 1].plot(epochs, train_psnr, 'g-', label='Training PSNR', linewidth=2)
    axes[1, 1].plot(epochs, val_psnr, 'orange', label='Validation PSNR', linewidth=2)
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs, train_ssim, 'purple', label='Training SSIM', linewidth=2, linestyle='--')
    ax2.plot(epochs, val_ssim, 'brown', label='Validation SSIM', linewidth=2, linestyle='--')
    
    axes[1, 1].set_title('Combined Accuracy Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PSNR (dB)', color='g')
    ax2.set_ylabel('SSIM', color='purple')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_metrics.pdf'), bbox_inches='tight')
    plt.show()
    
    # Individual plots for better clarity
    # Loss only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Profile', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_profile.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Accuracy only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_psnr, 'g-', label='Training PSNR', linewidth=2)
    plt.plot(epochs, val_psnr, 'orange', label='Validation PSNR', linewidth=2)
    plt.title('PSNR Accuracy Profile', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'accuracy_profile.png'), dpi=300, bbox_inches='tight')
    plt.show()


# === Save Training History ===
def save_training_history(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim, save_dir):
    """Save training history to CSV and JSON files"""
    
    # Create DataFrame
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_psnr': train_psnr,
        'val_psnr': val_psnr,
        'train_ssim': train_ssim,
        'val_ssim': val_ssim
    })
    
    # Save to CSV
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # Save to JSON
    history_dict = {
        'training_history': history_df.to_dict('records'),
        'best_metrics': {
            'best_val_loss': float(min(val_losses)),
            'best_val_psnr': float(max(val_psnr)),
            'best_val_ssim': float(max(val_ssim)),
            'best_epoch': int(val_losses.index(min(val_losses)) + 1)
        },
        'final_metrics': {
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'final_train_psnr': float(train_psnr[-1]),
            'final_val_psnr': float(val_psnr[-1]),
            'final_train_ssim': float(train_ssim[-1]),
            'final_val_ssim': float(val_ssim[-1])
        }
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    return history_df


def train(args):
    print("=== GPU Detection ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training on CPU will be very slow.")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            return

    # === Create output directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/kaggle/working/training_output_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Training outputs will be saved to: {save_dir}")

    # === Data Preparation ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = myDataset(args.dataset_dir, transform=transform)
    
    # Split dataset for training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device.type == 'cuda' else 4,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == 'cuda' else 4,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Dataset split: {train_size} training, {val_size} validation samples")

    print("=== Model Setup ===")
    model = CNN_ViT_UnderwaterRestorer().to(device)
    print(f"Model created and moved to {device}")

    # === Losses and Metrics ===
    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # === Training History Lists ===
    train_losses = []
    val_losses = []
    train_psnr = []
    val_psnr = []
    train_ssim = []
    val_ssim = []
    
    best_val_loss = float('inf')
    best_val_psnr = 0.0

    print("=== Starting Training ===")
    for epoch in range(args.epochs):
        # === Training Phase ===
        model.train()
        total_train_loss = 0
        total_train_psnr = 0
        total_train_ssim = 0
        num_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # === Forward Pass ===
            outputs = model(inputs)

            # === Combined Loss ===
            l1 = l1_loss_fn(outputs, targets)
            ssim_val = ssim_loss_fn(outputs, targets)
            ssim_loss = 1 - ssim_val
            color_loss = color_constancy_loss(outputs)
            loss = l1 + 0.2 * ssim_loss + 0.1 * color_loss

            # === Calculate Training Metrics ===
            psnr = calculate_psnr(outputs, targets)
            
            # === Backprop ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_psnr += psnr.item() if psnr != float('inf') else 40.0
            total_train_ssim += ssim_val.item()
            num_train_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr.item():.2f}',
                'ssim': f'{ssim_val.item():.3f}'
            })

            if batch_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # === Calculate Average Training Metrics ===
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_psnr = total_train_psnr / num_train_batches
        avg_train_ssim = total_train_ssim / num_train_batches
        
        # === Validation Phase ===
        avg_val_loss, avg_val_psnr, avg_val_ssim = validate(model, val_loader, device, l1_loss_fn, ssim_loss_fn)
        
        # === Update Learning Rate ===
        scheduler.step()
        
        # === Store History ===
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnr.append(avg_train_psnr)
        val_psnr.append(avg_val_psnr)
        train_ssim.append(avg_train_ssim)
        val_ssim.append(avg_val_ssim)

        # === Print Epoch Results ===
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}, SSIM: {avg_train_ssim:.3f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.3f}")

        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

        # === Save Best Models ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pth'))
            print(f"  ✓ Saved best model (loss): {best_val_loss:.4f}")
            
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_psnr.pth'))
            print(f"  ✓ Saved best model (PSNR): {best_val_psnr:.2f}")

    print("\n=== Training Completed ===")

    # === Save Final Models ===
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_weights.pth'))
    torch.save(model, os.path.join(save_dir, 'final_model_full.pth'))
    
    # Also save to working directory for compatibility
    torch.save(model.state_dict(), "/kaggle/working/underwater_model_weights.pth")
    torch.save(model, "/kaggle/working/underwater_model_full.pth")

    print(f"✓ Final models saved to {save_dir}")
    print("✓ Final models also saved to /kaggle/working/")

    # === Generate and Save Training Visualizations ===
    print("\n=== Generating Training Visualizations ===")
    save_training_plots(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim, save_dir)
    
    # === Save Training History ===
    history_df = save_training_history(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim, save_dir)
    
    # === Print Final Summary ===
    print("\n=== Training Summary ===")
    print(f"Best Validation Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    print(f"Best Validation PSNR: {max(val_psnr):.2f} (Epoch {val_psnr.index(max(val_psnr)) + 1})")
    print(f"Best Validation SSIM: {max(val_ssim):.3f} (Epoch {val_ssim.index(max(val_ssim)) + 1})")
    
    if device.type == 'cuda':
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
    
    print(f"\n✓ All training outputs saved to: {save_dir}")
    print("✓ Training graphs, metrics, and models are ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+ViT Underwater Restoration Model with Comprehensive Monitoring")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help='Path to combined UIEB + EVUP dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default="/kaggle/working/trained_model.pth")

    args = parser.parse_args()

    train(args)