import torch
import torch.nn as nn
import torch.nn.functional as F

"""
MultiHeadAttention Module
- Input txt shape: (B, N_seq, D_embed)
  - B: Batch size
  - N_seq: Number of sequences
  - N_heads: Number of attention heads
  - D_head: Dimension of a single head
  - D_embed: Dimension of word embeddings
"""

class MultiHeadAttention(nn.Module):  # 수정: nn.Module로 표기
    def __init__(self, D_embed, N_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.D_embed = D_embed
        self.N_heads = N_heads
        self.D_head = D_embed // N_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.D_head]))  # 수정: self.D_head 사용

        assert D_embed % N_heads == 0, "D_embed must be divisible by N_heads"

        self.w_q = nn.Linear(D_embed, D_embed)
        self.w_k = nn.Linear(D_embed, D_embed)
        self.w_v = nn.Linear(D_embed, D_embed)

    def forward(self, v_q, v_k, v_v, mask=None):
        """
        Forward Pass
        - v_q: Query tensor of shape (B, N_seq, D_embed)
        - v_k: Key tensor of shape (B, N_seq, D_embed)
        - v_v: Value tensor of shape (B, N_seq, D_embed)
        """
        B, N_seq, _ = v_q.shape

        # Linear projections
        q = self.w_q(v_q)  # q: (B, N_seq, D_embed)
        k = self.w_k(v_k)  # k: (B, N_seq, D_embed)
        v = self.w_v(v_v)  # v: (B, N_seq, D_embed)

        # Reshape for multi-head attention
        q = q.view(B, N_seq, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # q: (B, N_heads, N_seq, D_head)
        k = k.view(B, N_seq, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # k: (B, N_heads, N_seq, D_head)
        v = v.view(B, N_seq, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # v: (B, N_heads, N_seq, D_head)

        # Compute attention scores
        k_t = k.transpose(-2, -1)  # k_t: (B, N_heads, D_head, N_seq)
        attention_scores = torch.matmul(q, k_t) / self.scale  # attention_scores: (B, N_heads, N_seq, N_seq)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(self.mask == 0, float("-1e20"))

        # Compute attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)  # attention_probs: (B, N_heads, N_seq, N_seq)
        attention_probs = self.dropout(attention_probs)

        # Compute attention values
        attention_values = torch.matmul(attention_probs, v)  # attention_values: (B, N_heads, N_seq, D_head)

        # Concatenate heads and project back
        attention_values = attention_values.permute(0, 2, 1, 3).contiguous()  # attention_values: (B, N_seq, N_heads, D_head)
        attention_values = attention_values.view(B, N_seq, self.D_embed)  # attention_values: (B, N_seq, D_embed)

        return attention_values
    
    
    
import torch
import torch.nn as nn

class EncoderDecoderAttention(MultiHeadAttention):
    def __init__(self, D_decoder, D_encoder, D_q, N_heads, dropout=0.1):
        super(EncoderDecoderAttention, self).__init__(D_q, N_heads, dropout=dropout)
        self.w_q = nn.Linear(D_decoder, D_q)  # Query: From decoder input
        self.w_k = nn.Linear(D_encoder, D_q)  # Key: From encoder output
        self.w_v = nn.Linear(D_encoder, D_q)  # Value: From encoder output

    def forward(self, v_q, v_k, v_v, mask=None):
        """
        Forward Pass
        - v_q: Query tensor from decoder, shape (B, N_seq_decoder, D_decoder)
        - v_k: Key tensor from encoder, shape (B, N_seq_encoder, D_encoder)
        - v_v: Value tensor from encoder, shape (B, N_seq_encoder, D_encoder)
        - mask: Optional mask for attention
        """
        B = v_q.size(0)

        # Linear projections for Query, Key, Value
        q = self.w_q(v_q)  # Shape: (B, N_seq_decoder, D_q)
        k = self.w_k(v_k)  # Shape: (B, N_seq_encoder, D_q)
        v = self.w_v(v_v)  # Shape: (B, N_seq_encoder, D_q)

        # Reshape for multi-head attention
        q = q.view(B, -1, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # (B, N_heads, N_seq_decoder, D_head)
        k = k.view(B, -1, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # (B, N_heads, N_seq_encoder, D_head)
        v = v.view(B, -1, self.N_heads, self.D_head).permute(0, 2, 1, 3)  # (B, N_heads, N_seq_encoder, D_head)

        # Scaled Dot-Product Attention
        k_t = k.transpose(-2, -1)  # Transpose last two dimensions of K
        attention_scores = torch.matmul(q, k_t) / self.scale  # (B, N_heads, N_seq_decoder, N_seq_encoder)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e20'))

        attention_probs = torch.softmax(attention_scores, dim=-1)  # (B, N_heads, N_seq_decoder, N_seq_encoder)
        attention_probs = self.dropout(attention_probs)

        attention_values = torch.matmul(attention_probs, v)  # (B, N_heads, N_seq_decoder, D_head)

        # Concatenate heads
        attention_values = attention_values.permute(0, 2, 1, 3).contiguous()  # (B, N_seq_decoder, N_heads, D_head)
        attention_values = attention_values.view(B, -1, self.D_embed)  # (B, N_seq_decoder, D_q)

        return attention_values

class FFNN(nn.Module):
    def __init__(self,D_embed, dropout = 0.1):
        ## 
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(D_embed, D_embed*4)
        self.fc2 = nn.Linear(D_embed*4, D_embed)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out 
    
    
class ResConn(nn.Module):
    def __init__(self, D_embed, dropout = 0.1):
        super(ResConn, self).__init__()
        self.layer_norm = nn.LayerNorm(D_embed) # 
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x, sublayer_output):
        residual = x + self.dropout(sublayer_output)
        output = self.layer_norm(residual)
        return output
    
    
if __name__ == "__main__":
    # Text data example
    print("-" * 20 + " Text Data Example " + "-" * 20)
    D_embed = 32
    B = 20
    N_seq = 2
    N_heads = 4
    txt_data = torch.randn(B, N_seq, D_embed)  # Text data shape

    # Initialize MultiHeadAttention
    txtAttention = MultiHeadAttention(D_embed=D_embed, N_heads=N_heads)
    txt_output = txtAttention(txt_data, txt_data, txt_data)
    print(f"Text Attention Output Shape: {txt_output.shape}")  # Expected: (B, N_seq, D_embed)

    # Apply FFNN
    ffnn = FFNN(D_embed=D_embed)
    ffnn_output = ffnn(txt_output)
    print(f"Text FFNN Output Shape: {ffnn_output.shape}")  # Expected: (B, N_seq, D_embed)

    # Apply Residual Connection
    txt_res_conn = ResConn(D_embed=D_embed)
    txt_final_output = txt_res_conn(txt_data, ffnn_output)
    print(f"Text Residual Connection Output Shape: {txt_final_output.shape}")  # Expected: (B, N_seq, D_embed)

    # Image data example
    print("-" * 20 + " Image Data Example " + "-" * 20)
    img_B = 32
    img_D_image = 16
    img_D_vq = 8
    img_N_heads = 4
    img_data = torch.randn(img_B, img_D_image, img_D_vq)  # Image data shape

    # Initialize MultiHeadAttention
    imgAttention = MultiHeadAttention(D_embed=img_D_vq, N_heads=img_N_heads)
    img_output = imgAttention(img_data, img_data, img_data)
    print(f"Image Attention Output Shape: {img_output.shape}")  # Expected: (B, img_D_image, img_D_vq)

    # Apply FFNN
    img_ffnn = FFNN(D_embed=img_D_vq)
    img_ffnn_output = img_ffnn(img_output)
    print(f"Image FFNN Output Shape: {img_ffnn_output.shape}")  # Expected: (B, img_D_image, img_D_vq)

    # Apply Residual Connection
    img_res_conn = ResConn(D_embed=img_D_vq)
    img_final_output = img_res_conn(img_data, img_ffnn_output)
    print(f"Image Residual Connection Output Shape: {img_final_output.shape}")  # Expected: (B, img_D_image, img_D_vq)
    
    
    ### encoder decoder attention example
    D_decoder = 64
    D_encoder = 128
    D_q = 64
    N_heads = 8
    B = 32  # Batch size
    N_seq_decoder = 10  # Decoder sequence length
    N_seq_encoder = 15  # Encoder sequence length

    # Input tensors
    decoder_input = torch.randn(B, N_seq_decoder, D_decoder)  # Query from decoder
    encoder_output = torch.randn(B, N_seq_encoder, D_encoder)  # Key/Value from encoder

    # Initialize Encoder-Decoder Attention
    enc_dec_attention = EncoderDecoderAttention(D_decoder, D_encoder, D_q, N_heads)

    # Forward pass
    output = enc_dec_attention(v_q = decoder_input,v_k= encoder_output, v_v= encoder_output)
    print(f"Output Shape: {output.shape}")  # Expected: (B, N_seq_decoder, D_q)


        
        

