import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os

from model import CNN_ViT_UnderwaterRestorer
from dataset import myDataset  #  custom dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure



# === Color Constancy Loss Function ===
def color_constancy_loss(img):
    mean_rgb = torch.mean(img, dim=[2, 3], keepdim=True)
    mean_diff = torch.abs(mean_rgb - torch.mean(mean_rgb, dim=1, keepdim=True))
    return torch.mean(mean_diff)

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

    # === Data Preparation ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = myDataset(args.dataset_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device.type == 'cuda' else 4,
        pin_memory=(device.type == 'cuda')
    )

    print("=== Model Setup ===")
    model = CNN_ViT_UnderwaterRestorer().to(device)
    print(f"Model created and moved to {device}")

    # === Losses and Optimizer ===
    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_loss = float('inf')

    # === Optional: Mixed Precision ===
    # from torch.cuda.amp import GradScaler, autocast
    # scaler = GradScaler()

    print("=== Starting Training ===")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # === Forward Pass ===
            outputs = model(inputs)

            # === Combined Loss ===
            l1 = l1_loss_fn(outputs, targets)
            ssim = 1 - ssim_loss_fn(outputs, targets)
            color_loss = color_constancy_loss(outputs)
            loss = l1 + 0.2 * ssim + 0.1 * color_loss

            # === Backprop ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if batch_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        print(f"\n Epoch [{epoch + 1}/{args.epochs}] - Avg Loss: {avg_loss:.4f}")

        if device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
            print(f" Saved best model with loss: {best_loss:.4f} -> {args.save_path}")

    print(" Training finished.")

    torch.save(model.state_dict(), "/kaggle/working/underwater_model_weights.pth")
    torch.save(model, "/kaggle/working/underwater_model_full.pth")

    print(" Final models saved.")

    if device.type == 'cuda':
        print(f" Peak GPU memory usage: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+ViT Underwater Restoration Model")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help='Path to combined UIEB + EVUP dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default="/kaggle/working/trained_model.pth")

    args = parser.parse_args()

    train(args)
