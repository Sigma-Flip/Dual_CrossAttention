import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1, device='cpu'):
        super(MultiHeadAttention, self).__init__()
        self.D_embed = D_embed
        self.N_heads = N_heads
        self.D_head = D_embed // N_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.D_head])).to(device)
        self.device = device

        assert D_embed % N_heads == 0, "D_embed must be divisible by N_heads"

        self.w_q = nn.Linear(D_embed, D_embed).to(device)
        self.w_k = nn.Linear(D_embed, D_embed).to(device)
        self.w_v = nn.Linear(D_embed, D_embed).to(device)
        self.w_o = nn.Linear(D_embed, D_embed).to(device)  # Output projection layer

    def forward(self, v_q, v_k, v_v, mask=None):
        B, N_seq_q, _ = v_q.shape
        B, N_seq_k, _ = v_k.shape

        # Linear projections
        q = self.w_q(v_q).to(self.device)
        k = self.w_k(v_k).to(self.device)
        v = self.w_v(v_v).to(self.device)

        # Reshape for multi-head attention
        q = q.view(B, N_seq_q, self.N_heads, self.D_head).permute(0, 2, 1, 3)
        k = k.view(B, N_seq_k, self.N_heads, self.D_head).permute(0, 2, 1, 3)
        v = v.view(B, N_seq_k, self.N_heads, self.D_head).permute(0, 2, 1, 3)

        # Compute attention scores
        k_t = k.transpose(-2, -1)
        attention_scores = (torch.matmul(q, k_t) / self.scale).to(self.device)

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
        output = self.w_o(attention_values).to(self.device)

        return output


# Position-wise Feed-Forward Network
class FFNN(nn.Module):
    def __init__(self, D_embed, dropout=0.1, device='cpu'):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(D_embed, D_embed * 4).to(device)
        self.fc2 = nn.Linear(D_embed * 4, D_embed).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x):
        out = self.fc1(x).to(self.device)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out).to(self.device)
        return out

# Residual Connection
class ResConn(nn.Module):
    def __init__(self, D_embed, dropout=0.1, device='cpu'):
        super(ResConn, self).__init__()
        self.layer_norm = nn.LayerNorm(D_embed).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, x, sublayer_output):
        residual = x + self.dropout(sublayer_output)
        output = self.layer_norm(residual).to(self.device)
        return output

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, D_embed, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, D_embed, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, D_embed, 2, device=device).float() * (-math.log(10000.0) / D_embed))
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
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, device='cpu'):
        super(PositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width
        self.device = device

        # Create positional encodings
        pe = torch.zeros(3, height, width, device=device)
        y_position = torch.arange(0, height, device=device).unsqueeze(1).repeat(1, width).float()  # (height, width)
        x_position = torch.arange(0, width, device=device).unsqueeze(0).repeat(height, 1).float()  # (height, width)

        div_term = torch.exp(torch.arange(0, 2, 2, device=device).float() * (-math.log(10000.0) / 3))

        pe[0, :, :] = torch.sin(y_position * div_term[0])  # Y-axis positional encoding
        pe[1, :, :] = torch.cos(x_position * div_term[0])  # X-axis positional encoding
        pe[2, :, :] = torch.sin((y_position + x_position) * div_term[0])  # Combined positional encoding

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds 2D Positional Encoding to the input tensor.
        x: Input tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == 3, "Channel dimension must be 3."
        assert H == self.height and W == self.width, "Input dimensions must match positional encoding dimensions."
        pe_expanded = self.pe.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, C, H, W)
        return x + pe_expanded
class DecoderHead(nn.Module):
    def __init__(self, N_vocab: int, D_embed: int, N_heads: int, n_layers: int, device='cpu'):
        """
        DecoderHead: A simple decoder head for the text output.
        Args:
        - N_vocab: Vocabulary size
        - D_embed: Embedding dimension
        - N_heads: Number of attention heads
        - n_layers: Number of decoder layers
        """
        super(DecoderHead, self).__init__()  # Ensures proper initialization of nn.Module
        self.head = nn.Linear(D_embed, N_vocab).to(device)  # Final linear layer to map to vocabulary size

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Positional Encoding Test
    print("-" * 20 + " Positional Encoding Test " + "-" * 20)
    pos_encoder = PositionalEncoding(D_embed=D_embed, device=device)
    input_data = torch.randn(B, N_seq, D_embed, device=device)
    pos_encoded_data = pos_encoder(input_data)
    print(f"Positional Encoding Output Shape: {pos_encoded_data.shape}")  # Expected: (B, N_seq, D_embed)

    # Multi-Head Attention Test
    print("-" * 20 + " Multi-Head Attention Test " + "-" * 20)
    mha = MultiHeadAttention(D_embed=D_embed, N_heads=N_heads, device=device)
    attention_output = mha(pos_encoded_data, pos_encoded_data, pos_encoded_data)
    print(f"Attention Output Shape: {attention_output.shape}")  # Expected: (B, N_seq, D_embed)

    # FFNN Test
    print("-" * 20 + " Feed-Forward Network Test " + "-" * 20)
    ffnn = FFNN(D_embed=D_embed, device=device)
    ffnn_output = ffnn(attention_output)
    print(f"FFNN Output Shape: {ffnn_output.shape}")  # Expected: (B, N_seq, D_embed)

    # Residual Connection Test
    print("-" * 20 + " Residual Connection Test " + "-" * 20)
    res_conn = ResConn(D_embed=D_embed, device=device)
    final_output = res_conn(attention_output, ffnn_output)
    print(f"Residual Connection Output Shape: {final_output.shape}")  # Expected: (B, N_seq, D_embed)

    print("-" * 20 + "Image Network Test (including Positional Encoding) " + "-" * 20)

    B = 32  # Batch size
    height, width = 16, 16  # Original image dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print("-" * 20 + " Image Network Test (including Positional Encoding) " + "-" * 20)

    # Image data example
    image_data = torch.randn(B, 3, height, width, device=device)  # (B, 3, H, W)

    # Apply 2D Positional Encoding
    pos_enc_2d = PositionalEncoding2D(height=height, width=width, device=device)
    pos_encoded_image = pos_enc_2d(image_data)
    print(f"Positional Encoding Applied Shape: {pos_encoded_image.shape}")  # Expected: (B, 3, height, width)
