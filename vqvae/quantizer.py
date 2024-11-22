import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta):
        """
        Vector Quantizer for VQ-VAE
        - n_embeddings: Number of embeddings in the codebook
        - embedding_dim: Dimension of each embedding
        - beta: Commitment loss weight
        """
        super(VectorQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Codebook for embeddings
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

    def forward(self, z):
        """
        Forward pass for vector quantization.
        - Input: z of shape (B, N_seq_img, D_embed)
        - Output:
          - loss: Scalar commitment loss
          - quantized: Quantized tensor of shape (B, N_seq_img, D_embed)
          - perplexity: Codebook usage measure
        """
        # Flatten input
        B, N_seq_img, D_embed = z.shape
        z_flattened = z.contiguous().view(-1, D_embed)  # Ensure contiguous memory before view

        # Compute distances between z and embedding vectors
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )  # Shape: (B * N_seq_img, n_embeddings)

        # Find closest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # Shape: (B * N_seq_img, 1)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)  # Shape: (B * N_seq_img, n_embeddings)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)  # Shape: (B * N_seq_img, D_embed)
        quantized = quantized.view(B, N_seq_img, D_embed)  # Reshape to original shape

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        # Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return loss, quantized, perplexity, encodings, encoding_indices
