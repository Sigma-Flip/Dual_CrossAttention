import torch
import torch.nn as nn
from .core import MultiHeadAttention, FFNN, ResConn


class AttentionBlock(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1, device='cpu'):
        super(AttentionBlock, self).__init__()
        self.device = device
        self.attention = MultiHeadAttention(D_embed, N_heads, dropout=dropout, device=device)
        self.residual1 = ResConn(D_embed, dropout=dropout, device=device)  # Residual connection for attention
        self.FFNN = FFNN(D_embed, dropout=dropout, device=device)
        self.residual2 = ResConn(D_embed, dropout=dropout, device=device)  # Residual connection for FFNN

    def forward(self, input, mask=None):
        """
        Forward Pass
        - input: Input tensor of shape (B, N_seq, D_embed)
        - mask: Optional mask for Multi-Head Attention
        Returns:
        - output: Tensor of shape (B, N_seq, D_embed)
        """
        input = input.to(self.device)
        # Step 1: Multi-Head Attention
        attention_output = self.attention(input, input, input, mask)  # Self-attention uses input for q, k, and v

        # Step 2: Residual Connection + LayerNorm (Attention Block)
        attention_residual = self.residual1(input, attention_output)

        # Step 3: Feed-Forward Neural Network
        ffnn_output = self.FFNN(attention_residual)

        # Step 4: Residual Connection + LayerNorm (FFNN Block)
        output = self.residual2(attention_residual, ffnn_output)

        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1, device='cpu'):
        super(CrossAttentionBlock, self).__init__()
        self.device = device
        self.attention = MultiHeadAttention(D_embed, N_heads, dropout=dropout, device=device)
        self.residual1 = ResConn(D_embed, dropout=dropout, device=device)  # Residual connection for attention
        self.FFNN = FFNN(D_embed, dropout=dropout, device=device)
        self.residual2 = ResConn(D_embed, dropout=dropout, device=device)  # Residual connection for FFNN

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        Forward Pass
        - decoder_input: Query tensor of shape (B, N_seq_decoder, D_embed)
        - encoder_output: Key/Value tensor of shape (B, N_seq_encoder, D_embed)
        - mask: Optional mask for Multi-Head Attention
        Returns:
        - output: Tensor of shape (B, N_seq_decoder, D_embed)
        """
        decoder_input = decoder_input.to(self.device)
        encoder_output = encoder_output.to(self.device)
        # Step 1: Multi-Head Cross-Attention
        attention_output = self.attention(decoder_input, encoder_output, encoder_output, mask)

        # Step 2: Residual Connection + LayerNorm (Attention Block)
        attention_residual = self.residual1(decoder_input, attention_output)

        # Step 3: Feed-Forward Neural Network
        ffnn_output = self.FFNN(attention_residual)

        # Step 4: Residual Connection + LayerNorm (FFNN Block)
        output = self.residual2(attention_residual, ffnn_output)

        return output


class AttentionStack(nn.Module):
    def __init__(self, n: int, D_embed: int, N_heads: int, model=AttentionBlock, device='cpu'):
        super(AttentionStack, self).__init__()
        self.device = device
        self.stack = nn.ModuleList([model(D_embed, N_heads, device=device) for _ in range(n)])

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.stack:
            x = layer(x)
        return x


class CrossAttentionStack(nn.Module):
    def __init__(self, n: int, D_embed: int, N_heads: int, dropout=0.1, model=CrossAttentionBlock, device='cpu'):
        """
        CrossAttentionStack: Stacks multiple Cross-Attention Blocks
        Args:
        - n: Number of Cross-Attention Blocks
        - D_embed: Dimension of embeddings
        - N_heads: Number of attention heads
        - dropout: Dropout rate
        - device: Device to run the model on
        """
        super(CrossAttentionStack, self).__init__()
        self.device = device
        self.stack = nn.ModuleList([model(D_embed, N_heads, dropout=dropout, device=device) for _ in range(n)])

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        Forward Pass for CrossAttentionStack
        - decoder_input: Query tensor of shape (B, N_seq_decoder, D_embed)
        - encoder_output: Key/Value tensor of shape (B, N_seq_encoder, D_embed)
        - mask: Optional mask for attention
        Returns:
        - output: Tensor of shape (B, N_seq_decoder, D_embed)
        """
        decoder_input = decoder_input.to(self.device)
        encoder_output = encoder_output.to(self.device)
        for layer in self.stack:
            decoder_input = layer(decoder_input, encoder_output, mask)
        return decoder_input


if __name__ == "__main__":
    # 공통 설정
    D_embed = 16  # Embedding dimension
    N_heads = 8   # Number of heads
    B = 32        # Batch size
    N_seq_self = 10  # Sequence length for self-attention
    N_seq_decoder = 10  # Decoder sequence length
    N_seq_encoder = 15  # Encoder sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Self-Attention Block
    print("-" * 20 + " Self-Attention Block Example " + "-" * 20)
    self_input = torch.randn(B, N_seq_self, D_embed).to(device)  # Input for self-attention

    # Initialize Self-Attention Block
    self_attention_block = AttentionBlock(D_embed=D_embed, N_heads=N_heads, device=device)

    # Forward pass
    self_output = self_attention_block(self_input)
    print(f"Self-Attention Block Input Shape: {self_input.shape}")
    print(f"Self-Attention Block Output Shape: {self_output.shape}")  # Expected: (B, N_seq_self, D_embed)

    # Cross-Attention Block
    print("-" * 20 + " Cross-Attention Block Example " + "-" * 20)
    decoder_input = torch.randn(B, N_seq_decoder, D_embed).to(device)  # Query from decoder
    encoder_output = torch.randn(B, N_seq_encoder, D_embed).to(device)  # Key/Value from encoder

    # Initialize Cross-Attention Block
    cross_attention_block = CrossAttentionBlock(D_embed=D_embed, N_heads=N_heads, device=device)

    # Forward pass
    cross_output = cross_attention_block(decoder_input, encoder_output)
    print(f"Cross-Attention Block decoder input shape : {decoder_input.shape}")
    print(f"Cross-Attention Block Output Shape: {cross_output.shape}")  # Expected: (B, N_seq_decoder, D_embed)

    # Cross-Attention Stack
    B, N_seq_encoder, N_seq_decoder, D_embed = 4, 16, 8, 64
    N_heads = 8
    n_blocks = 3
    dropout = 0.1

    # Dummy Inputs
    decoder_input = torch.randn(B, N_seq_decoder, D_embed).to(device)  # (B, N_seq_decoder, D_embed)
    encoder_output = torch.randn(B, N_seq_encoder, D_embed).to(device)  # (B, N_seq_encoder, D_embed)

    # Cross-Attention Stack
    print("-" * 20 + " Cross-Attention Stack Example " + "-" * 20)

    cross_attention_stack = CrossAttentionStack(n=n_blocks, D_embed=D_embed, N_heads=N_heads, device=device)

    # Forward Pass
    output = cross_attention_stack(decoder_input, encoder_output)

    # Outputs
    print(f"Decoder Input Shape: {decoder_input.shape}")
    print(f"Encoder Output Shape: {encoder_output.shape}")
    print(f"Final Output Shape: {output.shape}")
