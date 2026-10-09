# ==============================================================================
# FortiTranM-FiLM: Two-Stage Training Pipeline
# Stage 1: Train foundational Transformer backbone (100k samples, 1000 max epochs, Early Stopping)
# Stage 2: Transfer Learning with FiLM (freeze backbone, train FiLM 50 epochs)
# Authors: Mahfoud Faraj Abdulmolla, Osama Abusaid
# ==============================================================================

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.fortitran_film import FortiTranM_FiLM


class OFDMChannelDataset(Dataset):
    """
    In-memory or memory-mapped dataset loader for pre-extracted channel realizations.
    """
    def __init__(self, data_path: str):
        print(f"Loading dataset from: {data_path}")
        cache = torch.load(data_path, map_location='cpu')
        
        # h_est: (N, 80) sparse pilot inputs (real and imag stacked or complex)
        # h_ideal: (N, 120, 14) ground truth channel matrices
        # conditions: (N, 3) metadata [SNR, DS, Doppler]
        self.h_est = cache['h_est']
        self.h_ideal = cache['h_ideal']
        self.conditions = cache.get('conditions', None)
        self.length = self.h_est.shape[0]
        print(f"Loaded {self.length} samples successfully.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.conditions is not None:
            return self.h_est[idx], self.h_ideal[idx], self.conditions[idx]
        return self.h_est[idx], self.h_ideal[idx], torch.zeros(3)


def train_stage1_backbone(model, train_loader, val_loader, device, max_epochs=1000, patience=12):
    """
    Stage 1: Train foundational hybrid CNN-Transformer without FiLM from scratch.
    """
    print("\n" + "="*70)
    print("STAGE 1: TRAINING FOUNDATIONAL TRANSFORMER BACKBONE")
    print(f"Configuration: Max {max_epochs} Epochs, Early Stopping Patience = {patience}")
    print("="*70)

    model.use_film = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for pilots, target, _ in train_loader:
            pilots, target = pilots.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(pilots)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * pilots.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pilots, target, _ in val_loader:
                pilots, target = pilots.to(device), target.to(device)
                output = model(pilots)
                loss = criterion(output, target)
                val_loss += loss.item() * pilots.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/backbone_best.pt')
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{max_epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} (Best: {best_val_loss:.5f} @ Ep {best_epoch})")

        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] Terminated at Epoch {epoch}. Best Epoch: {best_epoch} with Val MSE: {best_val_loss:.5f}")
            break

    model.load_state_dict(torch.load('checkpoints/backbone_best.pt'))
    return model


def train_stage2_film_transfer(model, train_loader, val_loader, device, transfer_epochs=50):
    """
    Stage 2: Transfer Learning with FiLM conditioning.
    Freeze backbone parameters and optimize FiLM MLP using channel state metadata.
    """
    print("\n" + "="*70)
    print("STAGE 2: TRANSFER LEARNING VIA FEATURE-WISE LINEAR MODULATION (FiLM)")
    print(f"Configuration: {transfer_epochs} Epochs, Backbone Frozen, Optimizing FiLM Layer")
    print("="*70)

    # 1. Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Enable gradients ONLY for FiLM block
    model.use_film = True
    for param in model.film_block.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters in FiLM Transfer Stage: {trainable_params:,} (Backbone weights frozen)")

    optimizer = torch.optim.Adam(model.film_block.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, transfer_epochs + 1):
        model.train()
        train_loss = 0.0

        for pilots, target, conditions in train_loader:
            pilots, target, conditions = pilots.to(device), target.to(device), conditions.to(device)
            optimizer.zero_grad()

            output = model(pilots, conditions)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.film_block.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * pilots.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pilots, target, conditions in val_loader:
                pilots, target, conditions = pilots.to(device), target.to(device), conditions.to(device)
                output = model(pilots, conditions)
                loss = criterion(output, target)
                val_loss += loss.item() * pilots.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/fortitran_film_final.pt')

        if epoch % 10 == 0 or epoch == 1:
            print(f"Transfer Epoch {epoch:2d}/{transfer_epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

    print("\nFiLM Transfer Learning successfully completed! Model saved to checkpoints/fortitran_film_final.pt")


def main():
    parser = argparse.ArgumentParser(description="FortiTranM-FiLM Training Suite")
    parser.add_argument('--train-data', type=str, default='data/train_100k.pt', help='Path to 100k train data')
    parser.add_argument('--val-data', type=str, default='data/val_10k.pt', help='Path to 10k val data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device(args.device)
    print(f"Execution Device: {device}")

    # Initialize model
    model = FortiTranM_FiLM(use_film=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total FortiTranM-FiLM Parameters: {total_params:,}")

    # Note: Replace paths with actual preprocessed .pt files when training locally
    if os.path.exists(args.train_data) and os.path.exists(args.val_data):
        train_ds = OFDMChannelDataset(args.train_data)
        val_ds = OFDMChannelDataset(args.val_data)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # Stage 1: Backbone
        model = train_stage1_backbone(model, train_loader, val_loader, device, max_epochs=1000, patience=12)

        # Stage 2: FiLM Transfer
        train_stage2_film_transfer(model, train_loader, val_loader, device, transfer_epochs=50)
    else:
        print(f"\n[Info] Data files '{args.train_data}' or '{args.val_data}' not found.")
        print("Please generate the dataset using the MATLAB scripts and run the pipeline.")


if __name__ == '__main__':
    main()
