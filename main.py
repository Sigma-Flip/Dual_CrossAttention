import torch
import torch.nn as nn
import torch.nn.functional as F
from vqvae.vqvae import VQVAE_Decoder, VQVAE_Encoder
from transformer import AttentionStack, CrossAttentionStack
from transformer import PositionalEncoding, PositionalEncoding2D
from transformer import DecoderHead
from transformers import AutoTokenizer


class ImgEncoder(nn.Module):
    def __init__(self, D_embed, H, W, # Positional Encoding params
                h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, # VQVAE params
                 N_stacks, N_heads): # Attention Stack params
        super(ImgEncoder, self).__init__()
        self.channel_projection = nn.Conv2d(in_channels=3, out_channels=D_embed, kernel_size=1)  # (B, C, H, W) -> (B, D_embed, H, W)
        self.positionalEncoding2d = PositionalEncoding2D(D_embed=D_embed, height=H, width=W)  # (B, D_embed, H, W)
        self.vqvaeEncoder = VQVAE_Encoder(D_embed, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta)  # (B, N_seq_img, D_embed)
        self.selfAttention = AttentionStack(N_stacks, D_embed, N_heads)  # (B, N_seq_img, D_embed) -> (B, N_seq_img, D_embed)
    
    def forward(self, img):
        # Project input channels to match D_embed
        img_projected = self.channel_projection(img)  # Shape: (B, D_embed, H, W)
        positioned_img = self.positionalEncoding2d(img_projected)  # Shape: (B, D_embed, H, W)
        embedding_loss, z_q, perplexity, z_e, encodings, encoding_indices = self.vqvaeEncoder(positioned_img)  # z_q: (B, N_seq_img, D_embed)
        attention_value = self.selfAttention(z_q)  # attention_value : (B, N_seq_img, D_embed)
        return attention_value, embedding_loss, perplexity, z_e, encodings, encoding_indices
    
class TxtEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str, N_stacks: int, D_embed: int, N_heads: int):
        """
        TxtEncoder: Encodes text sequences using self-attention.
        Args:
        - pretrained_model_name: Hugging Face pre-trained model name (e.g., 'bert-base-uncased')
        - N_stacks: Number of self-attention layers
        - D_embed: Embedding dimension
        - N_heads: Number of attention heads
        """
        super(TxtEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)  # Pre-trained tokenizer
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, D_embed)  # Embedding layer to convert input_ids to D_embed
        self.positionalEncoding1d = PositionalEncoding(D_embed)  # 1D Positional Encoding
        self.selfAttention = AttentionStack(N_stacks, D_embed, N_heads)  # Attention Stack

    def forward(self, txt):
        """
        Forward pass for TxtEncoder.
        Args:
        - txt: List of text sequences (e.g., ["hello world", "this is an example"])
        Returns:
        - attention_value: Output of self-attention stack
        """
        # Tokenize text and convert to embeddings
        tokenized = self.tokenizer(txt, return_tensors="pt", padding=True, truncation=True)  # input_ids: (B, N_seq)
        input_ids = tokenized["input_ids"]  # Shape: (B, N_seq)

        # Convert token IDs to embeddings using nn.Embedding
        embeddings = self.embedding(input_ids)  # Shape: (B, N_seq, D_embed)

        # Positional encoding and self-attention
        positioned_txt = self.positionalEncoding1d(embeddings)  # Shape: (B, N_seq, D_embed)
        attention_value = self.selfAttention(positioned_txt)  # Shape: (B, N_seq, D_embed)
        return attention_value

class CrossEncoder(nn.Module):
    def __init__(self, 
                D_embed, N_heads, dropout=0.1):
        super(CrossEncoder, self).__init__()
        self.crossEncoder = CrossAttentionStack(D_embed, N_heads, dropout=0.1)  # Cross-attention block
        
    def forward(self, decoder_input, encoder_output):
        '''
        decoder Input shape : (B, N_seq_decoder, D_embed)
        encoder Input shape : (B, N_seq_encoder, D_embed)
        Returns:
        - out: (B, N_seq_decoder, D_embed)
        '''
        out = self.crossEncoder(decoder_input, encoder_output) # out: (B, N_seq_decoder, D_embed)
        return out 
    
class ImgDecoder(nn.Module):
    def __init__(self,
                 embedding_dim, h_dim, n_res_layers, res_h_dim):
        super(ImgDecoder, self).__init__()
        self.imgDecoder = VQVAE_Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)  # (B, N_seq_img, D_embed)
    
    def forward(self, z_q):
        # Reshape sequence to 2D grid for decoding
        N_seq_img = z_q.shape[1]
        W_flat = int(N_seq_img**0.5)  # Assume square spatial dimension
        H_flat = W_flat
        img_reconstructed = self.imgDecoder(z_q, W_flat, H_flat)  # (B, C, H, W)
        return img_reconstructed
    
class TxtDecoder(nn.Module):
    def __init__(self,
                 N_vocab: int, D_embed: int, N_heads: int, n_layers: int):
        super(TxtDecoder, self).__init__()
        self.txtDecoder = DecoderHead(N_vocab, D_embed, N_heads, n_layers)  # (B, N_seq, D_embed) -> (B, N_seq, N_vocab)
    
    def forward(self, x):
        """
        Forward pass for TxtDecoder
        Args:
        - x: Input tensor of shape (B, N_seq, D_embed)
        Returns:
        - output: Tensor of shape (B, N_seq, N_vocab)
        """
        return self.txtDecoder(x)

# Main script
# Main script
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    print(f"device : {device}")  

    # Parameters
    B, C, W, H = 4, 3, 64, 64  # Image shape: Batch size, Channels, Width, Height
    N_vocab, D_embed, N_heads, N_stacks, n_layers = 30522, 64, 8, 3, 2  # Text-related parameters
    h_dim, res_h_dim, n_res_layers, n_embeddings, beta = 128, 64, 3, 512, 0.25  # VQ-VAE parameters

    # Example input data
    img_data = torch.randn(B, C, W, H).to(device)  # Random image tensor moved to device
    txt_data = ["Hello world", "This is a test", "Transformer example", "PyTorch integration"]

    # Initialize models and move them to the selected device
    img_encoder = ImgEncoder(D_embed=D_embed, H=W, W=H, h_dim=h_dim, res_h_dim=res_h_dim,
                             n_res_layers=n_res_layers, n_embeddings=n_embeddings,
                             embedding_dim=D_embed, beta=beta, N_stacks=N_stacks, N_heads=N_heads).to(device)
    
    txt_encoder = TxtEncoder(pretrained_model_name="bert-base-uncased",
                             N_stacks=N_stacks, D_embed=D_embed, N_heads=N_heads).to(device)
    
    cross_encoder = CrossEncoder(D_embed=D_embed, N_heads=N_heads).to(device)
    
    img_decoder = ImgDecoder(embedding_dim=D_embed, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim).to(device)
    
    txt_decoder = TxtDecoder(N_vocab=N_vocab, D_embed=D_embed, N_heads=N_heads, n_layers=n_layers).to(device)

    # Forward pass for image
    img_attention_value, img_embedding_loss, img_perplexity, z_e, _, _ = img_encoder(img_data)  # img data : (B,C,W,H)

    # Forward pass for text
    txt_attention_value = txt_encoder(txt_data)  # txt_data : (B, N_seq)
    txt_attention_value = txt_attention_value.to(device)  # Moving output to device

    # Cross-attention
    img2txt_encoded_txt = cross_encoder(decoder_input=txt_attention_value, encoder_output=img_attention_value)
    txt2img_encoded_img = cross_encoder(decoder_input=img_attention_value, encoder_output=txt_attention_value)

    # Decode image and text
    reconstructed_img = img_decoder(txt2img_encoded_img)  # (B, C, W, H)
    reconstructed_txt = txt_decoder(img2txt_encoded_txt)  # (B, N_vocab)

    # Outputs
    print("*" * 20, "Results", "*" * 20)
    print(f"Reconstructed Image Shape: {reconstructed_img.shape}")  # Expected: (B, C, W, H)
    print(f"Reconstructed Text Shape: {reconstructed_txt.shape}")  # Expected: (B, N_vocab)
    print(f"Image Embedding Loss: {img_embedding_loss.item()}")
    print(f"Image Perplexity: {img_perplexity.item()}")
