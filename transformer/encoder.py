# from transformer.core import MultiHeadAttention, FFNN, ResConn
# import torch
# import torch.nn as nn

# class SelfAttention(nn.Module):
#     def __init__(self, D_embed, N_heads, dropout=0.1):
#         super(SelfAttention, self).__init__()
#         self.S_attention = MultiHeadAttention(D_embed, N_heads, dropout=dropout) # self attention
#         self.residual1 = ResConn(D_embed, dropout=dropout)  # Residual connection for attention
#         self.FFNN = FFNN(D_embed, dropout=dropout)
#         self.residual2 = ResConn(D_embed, dropout=dropout)  # Residual connection for FFNN

#     def forward(self, input, mask=None):
#         """
#         Forward Pass
#         - input: Input tensor of shape (B, N_seq, D_embed)
#         - mask: Optional mask for Multi-Head Attention
#         Returns:
#         - output: Tensor of shape (B, N_seq, D_embed)
#         """
#         # Step 1: Multi-Head Attention
#         attention_output = self.S_attention(input, input, input)  # Self-attention uses input for q, k, and v

#         # Step 2: Residual Connection + LayerNorm (Attention Block)
#         attention_residual = self.residual1(input, attention_output)

#         # Step 3: Feed-Forward Neural Network
#         ffnn_output = self.FFNN(attention_residual)

#         # Step 4: Residual Connection + LayerNorm (FFNN Block)
#         output = self.residual2(attention_residual, ffnn_output)

#         return output
    

    

        
    
    
# if __name__ == "__main__":
#     print("-" * 20 + " Text Data Example with SelfAttention " + "-" * 20)
#     D_embed = 32
#     B = 20
#     N_seq = 10
#     N_heads = 4

#     # Example text data
#     txt_data = torch.randn(B, N_seq, D_embed)  # Text data shape (B, N_seq, D_embed)

#     # Initialize SelfAttention
#     txt_sa = SelfAttention(D_embed, N_heads)
#     sa_output = txt_sa(txt_data)
#     print(f"SelfAttention Output Shape: {sa_output.shape}")  # Expected: (B, N_seq, D_embed)

#     print("-" * 20 + " CrossAttention Example " + "-" * 20)
#      ### encoder decoder attention example
#     D_decoder = 64
#     D_encoder = 128
#     D_q = 64
#     N_heads = 8
#     B = 32  # Batch size
#     N_seq_decoder = 10  # Decoder sequence length
#     N_seq_encoder = 15  # Encoder sequence length

#     # Input tensors
#     decoder_input = torch.randn(B, N_seq_decoder, D_decoder)  # Query from decoder
#     encoder_output = torch.randn(B, N_seq_encoder, D_encoder)  # Key/Value from encoder

#     # Initialize Encoder-Decoder Attention
#     enc_dec_attention = CrossAttention(D_decoder, D_encoder, D_q, N_heads)

#     # Forward pass
#     output = enc_dec_attention(decoder_input=decoder_input,encoder_output= encoder_output)
#     print(f"Cross attention Output Shape: {output.shape}")  # Expected: (B, N_seq_decoder, D_q)

    
#     # apply cross_ffnn
#     cross_ffnn = FFNN(D_embed=D_decoder)
#     cross_ffnn_output = cross_ffnn(output)
#     print(f"Cross FFNN Output Shape: {cross_ffnn_output.shape}")  # Expected: (B, N_seq_decoder, D_decoder)
    
#     # apopply Residual Connection
#     cross_res_conn = ResConn(D_embed=D_decoder)
#     cross_final_output = cross_res_conn(decoder_input, cross_ffnn_output)
#     print(f"Cross FFNN Output Shape: {cross_final_output.shape}")  # Expected: (B, N_seq_decoder, D_decoder)

    
