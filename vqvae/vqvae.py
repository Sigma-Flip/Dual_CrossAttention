
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .quantizer import VectorQuantizer
from .decoder import Decoder
from .encoder import Encoder

class VQVAE_Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        """
        VQVAE Encoder
        - h_dim: Hidden dimension of encoder
        - res_h_dim: Residual block dimension
        - n_res_layers: Number of residual layers
        - n_embeddings: Number of embeddings in vector quantizer
        - embedding_dim: Dimension of each embedding (D_embed)
        - beta: Commitment loss weight
        """
        super(VQVAE_Encoder, self).__init__()
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, embedding_dim)
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            kernel_size=1, 
            stride=1
        )
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

    def forward(self, x):
        """
        Forward pass for VQVAE Encoder
        - Input: x, shape (B, C, W, H)
        - Output: embedding_loss, z_q, perplexity, z_e, encodings, encoding_indices
        """
        z_e = self.encoder(x)  # (B, N_seq_img, D_embed)

        # Apply pre-quantization convolution
        z_e = z_e.permute(0, 2, 1)  # Reshape to (B, D_embed, N_seq_img)
        z_e = self.pre_quantization_conv(z_e)  # (B, D_embed, N_seq_img)
        z_e = z_e.permute(0, 2, 1)  # Back to (B, N_seq_img, D_embed)

        # Vector quantization
        embedding_loss, z_q, perplexity, encodings, encoding_indices = self.vector_quantization(z_e)
        return embedding_loss, z_q, perplexity, z_e, encodings, encoding_indices



class VQVAE_Decoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, n_res_layers, res_h_dim):
        super(VQVAE_Decoder, self).__init__()
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, z_q, W, H):
        # Reshape to (B, D_embed, W', H') for decoding
        B, N_seq_img, D_embed = z_q.shape
        z_q = z_q.view(B, W, H, D_embed).permute(0, 3, 1, 2)  # (B, D_embed, W', H')

        # Decode
        return self.decoder(z_q)


if __name__ == "__main__":
    # Parameters
    B, C, W, H = 4, 3, 64, 64  # Input shape
    h_dim, res_h_dim, n_res_layers, embed_dim = 128, 64, 3, 64
    n_embeddings, beta = 512, 0.25

    # Input tensor
    x = torch.randn(B, C, W, H)

    # Initialize Encoder and Decoder
    encoder = VQVAE_Encoder(h_dim, res_h_dim, n_res_layers, n_embeddings, embed_dim, beta)
    decoder = VQVAE_Decoder(embedding_dim=embed_dim, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim)

    # Forward pass through Encoder
    embedding_loss, z_q, perplexity, z_e, _, _ = encoder(x)

    # Compute W_flat and H_flat (based on N_seq_img = W' * H')
    N_seq_img = z_q.shape[1]
    W_flat = int(N_seq_img**0.5)  # Assume square spatial dimension
    H_flat = W_flat

    # Forward pass through Decoder
    x_reconstructed = decoder(z_q, W_flat, H_flat)

    # Compute Reconstruction Loss
    reconstruction_loss = F.mse_loss(x_reconstructed, x)

    # Print Outputs
    print(f"B : {B}, C: {C}, W: {W}, H: {H}")
    print("*"*20, "SHAPE", "*"*20)
    print("Input shape:", x.shape)  # (B, C, W, H)
    print("Encoded shape (z_e):", z_e.shape)  # (B, N_seq_img, D_embed)
    print("Quantized shape (z_q):", z_q.shape)  # (B, N_seq_img, D_embed)
    print("Reconstructed shape:", x_reconstructed.shape)  # (B, C, W, H)
    # print("Quantized input exampleL:", z_q)
    print("*"*20, "LOSS", "*"*20)
    print("Embedding Loss:", embedding_loss.item())
    print("Perplexity:", perplexity.item())
    print("Reconstruction Loss:", reconstruction_loss.item())

