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

# Parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
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

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_embedding_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_perplexity = 0.0

        # Using tqdm to show progress bar for batch iteration
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
                total_loss.backward()  # Compute gradients
                optimizer.step()  # Update parameters

                # Metrics collection
                epoch_loss += total_loss.item()
                epoch_embedding_loss += embedding_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()
                epoch_perplexity += perplexity.item()

                # Update tqdm progress bar with loss values
                tepoch.set_postfix({
                    "Total Loss": total_loss.item(),
                    "Reconstruction Loss": reconstruction_loss.item(),
                    "Embedding Loss": embedding_loss.item(),
                    "Perplexity": perplexity.item()
                })

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_embedding_loss = epoch_embedding_loss / len(train_loader)
        avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
        avg_perplexity = epoch_perplexity / len(train_loader)

        loss_curve.append(avg_loss)
        embedding_loss_curve.append(avg_embedding_loss)
        reconstruction_loss_curve.append(avg_reconstruction_loss)
        perplexity_curve.append(avg_perplexity)

        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}, "
              f"Reconstruction Loss: {avg_reconstruction_loss:.4f}, "
              f"Embedding Loss: {avg_embedding_loss:.4f}, "
              f"Perplexity: {avg_perplexity:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    # Plot Loss Curves
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, EPOCHS + 1), loss_curve, label="Total Loss")
    plt.plot(range(1, EPOCHS + 1), embedding_loss_curve, label="Embedding Loss")
    plt.plot(range(1, EPOCHS + 1), reconstruction_loss_curve, label="Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curves.png"))
    plt.show()

    # Plot Perplexity Curve
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, EPOCHS + 1), perplexity_curve, label="Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, "perplexity_curve.png"))
    plt.show()
