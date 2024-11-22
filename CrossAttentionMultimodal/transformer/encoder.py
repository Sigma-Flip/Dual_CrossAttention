from attention import MultiHeadAttention, CrossAttention, FFNN, ResConn
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(D_embed, N_heads, dropout=dropout)
        self.residual1 = ResConn(D_embed, dropout=dropout)  # Residual connection for attention
        self.FFNN = FFNN(D_embed, dropout=dropout)
        self.residual2 = ResConn(D_embed, dropout=dropout)  # Residual connection for FFNN

    def forward(self, input, mask=None):
        """
        Forward Pass
        - input: Input tensor of shape (B, N_seq, D_embed)
        - mask: Optional mask for Multi-Head Attention
        Returns:
        - output: Tensor of shape (B, N_seq, D_embed)
        """
        # Step 1: Multi-Head Attention
        attention_output = self.attention(input, input, input)  # Self-attention uses input for q, k, and v

        # Step 2: Residual Connection + LayerNorm (Attention Block)
        attention_residual = self.residual1(input, attention_output)

        # Step 3: Feed-Forward Neural Network
        ffnn_output = self.FFNN(attention_residual)

        # Step 4: Residual Connection + LayerNorm (FFNN Block)
        output = self.residual2(attention_residual, ffnn_output)

        return output
    

    
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        
    
    
    
    
    
    
    
if __name__ == "__main__":
    print("-" * 20 + " Text Data Example " + "-" * 20)
    D_embed = 32
    B = 20
    N_seq = 2
    N_heads = 4
    txt_data = torch.randn(B, N_seq, D_embed)  # Text data shape

    # Initialize MultiHeadAttention
    txt_sa = SelfAttention(D_embed, N_heads)
    output = txt_sa(txt_data)
    print(output.shape) # expected : (B, N-seq, D-embed)
