import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Multi-Head Attention Class
# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.D_embed = D_embed
        self.N_heads = N_heads
        self.D_head = D_embed // N_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.D_head]))

        assert D_embed % N_heads == 0, "D_embed must be divisible by N_heads"

        self.w_q = nn.Linear(D_embed, D_embed)
        self.w_k = nn.Linear(D_embed, D_embed)
        self.w_v = nn.Linear(D_embed, D_embed)
        self.w_o = nn.Linear(D_embed, D_embed)  # Output projection layer

    def forward(self, v_q, v_k, v_v, mask=None):
        B, N_seq_q, _ = v_q.shape
        B, N_seq_k, _ = v_k.shape

        # Linear projections
        q = self.w_q(v_q)
        k = self.w_k(v_k)
        v = self.w_v(v_v)

        # Reshape for multi-head attention
        q = q.view(B, N_seq_q, self.N_heads, self.D_head).permute(0, 2, 1, 3)
        k = k.view(B, N_seq_k, self.N_heads, self.D_head).permute(0, 2, 1, 3)
        v = v.view(B, N_seq_k, self.N_heads, self.D_head).permute(0, 2, 1, 3)

        # Compute attention scores
        k_t = k.transpose(-2, -1)
        attention_scores = torch.matmul(q, k_t) / self.scale

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        # Compute attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute attention values
        attention_values = torch.matmul(attention_probs, v)
        attention_values = attention_values.permute(0, 2, 1, 3).contiguous()
        attention_values = attention_values.view(B, N_seq_q, self.D_embed)

        # Apply output projection
        output = self.w_o(attention_values)

        return output


# Position-wise Feed-Forward Network
class FFNN(nn.Module):
    def __init__(self, D_embed, dropout=0.1):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(D_embed, D_embed * 4)
        self.fc2 = nn.Linear(D_embed * 4, D_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Residual Connection
class ResConn(nn.Module):
    def __init__(self, D_embed, dropout=0.1):
        super(ResConn, self).__init__()
        self.layer_norm = nn.LayerNorm(D_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer_output):
        residual = x + self.dropout(sublayer_output)
        output = self.layer_norm(residual)
        return output

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, D_embed, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, D_embed)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, D_embed, 2).float() * (-math.log(10000.0) / D_embed))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension (1, max_len, D_embed)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        x: (B, N_seq, D_embed)
        """
        x = x + self.pe[:, :x.size(1)]
        return x
    
# Positional Encoding2D Class
class PositionalEncoding2D(nn.Module):
    def __init__(self, D_embed, height, width):
        super(PositionalEncoding2D, self).__init__()
        assert D_embed % 2 == 0, "D_embed must be divisible by 2 for 2D Positional Encoding."
        self.D_embed = D_embed
        self.height = height
        self.width = width

        # Create positional encodings
        pe = torch.zeros(D_embed, height, width)
        y_position = torch.arange(0, height).unsqueeze(1).repeat(1, width).float()  # (height, width)
        x_position = torch.arange(0, width).unsqueeze(0).repeat(height, 1).float()  # (height, width)

        div_term = torch.exp(torch.arange(0, D_embed, 2).float() * (-math.log(10000.0) / D_embed))

        pe[0::2, :, :] = torch.sin(y_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # Even indices
        pe[1::2, :, :] = torch.cos(x_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))  # Odd indices

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds 2D Positional Encoding to the input tensor.
        x: Input tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.height and W == self.width, "Input dimensions must match positional encoding dimensions."
        pe_expanded = self.pe.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, D_embed, H, W)
        return x + pe_expanded

class DecoderHead(nn.Module):
    def __init__(self, N_vocab: int, D_embed: int, N_heads: int, n_layers: int):
        """
        DecoderHead: A simple decoder head for the text output.
        Args:
        - N_vocab: Vocabulary size
        - D_embed: Embedding dimension
        - N_heads: Number of attention heads
        - n_layers: Number of decoder layers
        """
        super(DecoderHead, self).__init__()  # Ensures proper initialization of nn.Module
        self.head = nn.Linear(D_embed, N_vocab)  # Final linear layer to map to vocabulary size

    def forward(self, x):
        """
        Forward pass for DecoderHead
        Args:
        - x: Input tensor of shape (B, N_seq, D_embed)
        Returns:
        - output: Tensor of shape (B, N_seq, N_vocab)
        """
        return self.head(x)
    
# Test Model Integration
if __name__ == "__main__":
    D_embed = 64
    N_heads = 8
    N_seq = 10
    B = 32

    # Positional Encoding Test
    print("-" * 20 + " Positional Encoding Test " + "-" * 20)
    pos_encoder = PositionalEncoding(D_embed=D_embed)
    input_data = torch.randn(B, N_seq, D_embed)
    pos_encoded_data = pos_encoder(input_data)
    print(f"Positional Encoding Output Shape: {pos_encoded_data.shape}")  # Expected: (B, N_seq, D_embed)

    # Multi-Head Attention Test
    print("-" * 20 + " Multi-Head Attention Test " + "-" * 20)
    mha = MultiHeadAttention(D_embed=D_embed, N_heads=N_heads)
    attention_output = mha(pos_encoded_data, pos_encoded_data, pos_encoded_data)
    print(f"Attention Output Shape: {attention_output.shape}")  # Expected: (B, N_seq, D_embed)

    # FFNN Test
    print("-" * 20 + " Feed-Forward Network Test " + "-" * 20)
    ffnn = FFNN(D_embed=D_embed)
    ffnn_output = ffnn(attention_output)
    print(f"FFNN Output Shape: {ffnn_output.shape}")  # Expected: (B, N_seq, D_embed)

    # Residual Connection Test
    print("-" * 20 + " Residual Connection Test " + "-" * 20)
    res_conn = ResConn(D_embed=D_embed)
    final_output = res_conn(attention_output, ffnn_output)
    print(f"Residual Connection Output Shape: {final_output.shape}")  # Expected: (B, N_seq, D_embed)

    print("-" * 20 + "Image Network Test (including Positional Encoding) " + "-" * 20)

    # Input image dimensions
    B = 32  # Batch size
    height, width = 16, 16  # Original image dimensions
    D_embed = 64  # Embedding dimension

    # Image data example
    image_data = torch.randn(B, D_embed, height, width)  # (B, D_embed, H, W)

    # Apply 2D Positional Encoding
    pos_enc_2d = PositionalEncoding2D(D_embed=D_embed, height=height, width=width)
    pos_encoded_image = pos_enc_2d(image_data)
    print(f"Positional Encoding Applied Shape: {pos_encoded_image.shape}")  # Expected: (B, D_embed, height, width)
