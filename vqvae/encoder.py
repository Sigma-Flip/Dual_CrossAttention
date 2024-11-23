import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualStack

class Encoder(nn.Module):
    """
    Modified Encoder
    - Input shape: (B, C, W, H)
    - Output shape: (B, N_seq_img, D_embed), where N_seq_img = Flattened (W', H')
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, D_embed):
        """
        Args:
        - in_dim: Input channels (e.g., 3 for RGB images)
        - h_dim: Hidden dimension after Conv layers
        - n_res_layers: Number of residual layers
        - res_h_dim: Residual block hidden dimension
        - D_embed: Final embedding dimension (D_embed)
        """
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Conv2d(h_dim, D_embed, kernel_size=3, stride=1, padding=1)  # Final embedding dimension
        )

    def forward(self, x):
        # Convolution output: (B, D_embed, W', H')
        x = self.conv_stack(x)

        # Flatten spatial dimensions: (B, D_embed, W', H') -> (B, N_seq_img, D_embed)
        B, D_embed, W, H = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, W, H, D_embed)
        x = x.view(B, -1, D_embed)  # (B, N_seq_img, D_embed)

        return x
if __name__ == "__main__":
    # Random input data
    B, C, W, H = 4, 3, 64, 64  # Batch size, Channels, Width, Height
    x = torch.randn(B, C, W, H)

    # Initialize encoder
    encoder = Encoder(in_dim=3, h_dim=128, n_res_layers=3, res_h_dim=64, D_embed=64)

    # Forward pass
    encoder_out = encoder(x)

    print("Input shape:", x.shape)  # (B, C, W, H)
    print("Encoder output shape:", encoder_out.shape)  # (B, N_seq_img, D_embed)
