import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vqvae.vqvae import VQVAE_Encoder, VQVAE_Decoder
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Progress bar
import numpy as np

# Parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if __name__ == "__main__":  # Main block for multiprocessing compatibility
    # Enable anomaly detection to help debug gradient-related issues
    torch.autograd.set_detect_anomaly(True)

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize VQVAE Encoder and Decoder
    h_dim, res_h_dim, n_res_layers, embed_dim = 128, 64, 3, 64
    n_embeddings, beta = 512, 0.25

    encoder = VQVAE_Encoder(in_dim=3, h_dim=h_dim, res_h_dim=res_h_dim,
                            n_res_layers=n_res_layers, n_embeddings=n_embeddings,
                            embedding_dim=embed_dim, beta=beta).to(DEVICE)

    decoder = VQVAE_Decoder(embedding_dim=embed_dim, h_dim=h_dim,
                            n_res_layers=n_res_layers, res_h_dim=res_h_dim).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    # Training Loop
    best_loss = float("inf")
    loss_curve = []
    embedding_loss_curve = []
    reconstruction_loss_curve = []
    perplexity_curve = []
    patience_counter = 0
    

import numpy as np

# Updated Training Loop
best_losses = [float("inf")] * 5  # Top 5 losses
best_models = [None] * 5  # Top 5 model checkpoints

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()

    epoch_loss = 0.0
    epoch_embedding_loss = 0.0
    epoch_reconstruction_loss = 0.0
    epoch_perplexity = 0.0

    # Training loop with progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch [{epoch}/{EPOCHS}]")

        for batch_idx, (data, _) in enumerate(tepoch):
            data = data.to(DEVICE)  # Shape: (B, C, W, H)

            # Forward pass through Encoder and Decoder
            embedding_loss, z_q, perplexity, z_e, _, _ = encoder(data)
            N_seq_img = z_q.shape[1]
            W_flat = int(N_seq_img ** 0.5)
            H_flat = W_flat
            reconstructed_data = decoder(z_q, W_flat, H_flat)

            # Compute losses
            reconstruction_loss = F.mse_loss(reconstructed_data, data)
            total_loss = reconstruction_loss + embedding_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Metrics collection
            epoch_loss += total_loss.item()
            epoch_embedding_loss += embedding_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_perplexity += perplexity.item()

            # Update progress bar
            tepoch.set_postfix({
                "Total Loss": total_loss.item(),
                "Reconstruction Loss": reconstruction_loss.item(),
                "Embedding Loss": embedding_loss.item(),
                "Perplexity": perplexity.item()
            })

    # Calculate average metrics for the epoch
    avg_loss = epoch_loss / len(train_loader)
    avg_embedding_loss = epoch_embedding_loss / len(train_loader)
    avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
    avg_perplexity = epoch_perplexity / len(train_loader)

    # Save metrics to curves
    loss_curve.append(avg_loss)
    embedding_loss_curve.append(avg_embedding_loss)
    reconstruction_loss_curve.append(avg_reconstruction_loss)
    perplexity_curve.append(avg_perplexity)

    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}, "
          f"Reconstruction Loss: {avg_reconstruction_loss:.4f}, "
          f"Embedding Loss: {avg_embedding_loss:.4f}, "
          f"Perplexity: {avg_perplexity:.4f}")

    # Save the epoch model
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": avg_loss,
    }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_model.pth"))

    # Check and update best losses
    if avg_loss < max(best_losses):
        idx = best_losses.index(max(best_losses))
        best_losses[idx] = avg_loss
        best_models[idx] = {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
        }

    # Save test reconstructions every 10 epochs
    if epoch:
        encoder.eval()
        decoder.eval()
        test_data = next(iter(train_loader))[0][:30].to(DEVICE)  # Select first 30 images from train loader
        _, z_q, _, _, _, _ = encoder(test_data)
        N_seq_img = z_q.shape[1]
        W_flat, H_flat = int(N_seq_img**0.5), int(N_seq_img**0.5)
        reconstructed_test_data = decoder(z_q, W_flat, H_flat).cpu().detach()

        fig, axs = plt.subplots(2, 15, figsize=(15, 2))
        for i in range(15):
            axs[0, i].imshow(test_data[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            axs[1, i].imshow(reconstructed_test_data[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            axs[0, i].axis('off')
            axs[1, i].axis('off')
        plt.suptitle(f"Reconstruction Examples - Epoch {epoch}")
        plt.savefig(os.path.join(CHECKPOINT_DIR, f"reconstruction_epoch_{epoch}.png"))
        plt.close()

# Save the top 5 models
for i, model in enumerate(best_models):
    torch.save(model, os.path.join(CHECKPOINT_DIR, f"best_model_{i+1}.pth"))

# Plot Loss Curves
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(loss_curve) + 1), loss_curve, label="Total Loss")
plt.plot(range(1, len(embedding_loss_curve) + 1), embedding_loss_curve, label="Embedding Loss")
plt.plot(range(1, len(reconstruction_loss_curve) + 1), reconstruction_loss_curve, label="Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curves_combined.png"))
plt.show()
