import torch
import torch.nn as nn
import torch.nn.functional as F
from vqvae.vqvae import VQVAE_Decoder, VQVAE_Encoder
from transformer import AttentionStack, CrossAttentionStack
from transformer import PositionalEncoding, PositionalEncoding2D
from transformer import DecoderHead
from transformers import AutoTokenizer


class ImgEncoder(nn.Module):
    def __init__(self, H, W,  # 위치 인코딩 파라미터
                 h_dim, res_h_dim, n_res_layers, n_embeddings, D_embed, beta,  # VQVAE 파라미터
                 N_stacks, N_heads, device='cpu'):  # 어텐션 스택 파라미터
        super(ImgEncoder, self).__init__()
        self.device = device

        # VQVAE 인코더의 기대에 맞춰 입력 채널 수를 3에서 h_dim으로 변환합니다.
        self.positionalEncoding2d = PositionalEncoding2D(height=H, width=W, device=device)  # 위치 인코딩

        # VQVAE 인코더 정의 (입력 채널 수는 3)
        self.vqvaeEncoder = VQVAE_Encoder(3, h_dim, res_h_dim, n_res_layers, n_embeddings, D_embed, beta).to(device)  # (B, h_dim, H, W)

        # 이미지 표현을 위한 어텐션 스택
        self.selfAttention = AttentionStack(N_stacks, D_embed, N_heads, device=device)  # (B, N_seq_img, embedding_dim) -> (B, N_seq_img, embedding_dim)

    def forward(self, img):
        img = img.to(self.device)
        # 위치 인코딩 - 채널 프로젝션 이후 적용하여 입력 차원 유지
        positioned_img = self.positionalEncoding2d(img)  # Shape: (B, 3, H, W)

        # VQVAE 인코더에 통과시킵니다.
        embedding_loss, z_q, perplexity, z_e, encodings, encoding_indices = self.vqvaeEncoder(positioned_img)  # (B, N_seq_img, embedding_dim)

        # VQVAE 인코더 이후 어텐션 스택 적용
        attention_value = self.selfAttention(z_q)  # attention_value: (B, N_seq_img, embedding_dim)

        return attention_value, embedding_loss, perplexity, z_e, encodings, encoding_indices

    
class TxtEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str, N_stacks: int, D_embed: int, N_heads: int, device='cpu'):
        """
        TxtEncoder: Encodes text sequences using self-attention.
        Args:
        - pretrained_model_name: Hugging Face pre-trained model name (e.g., 'bert-base-uncased')
        - N_stacks: Number of self-attention layers
        - D_embed: Embedding dimension
        - N_heads: Number of attention heads
        """
        super(TxtEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)  # Pre-trained tokenizer
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, D_embed).to(device)  # Embedding layer to convert input_ids to D_embed
        self.positionalEncoding1d = PositionalEncoding(D_embed, device=device)  # 1D Positional Encoding
        self.selfAttention = AttentionStack(N_stacks, D_embed, N_heads, device=device)  # Attention Stack

    def forward(self, txt):
        """
        Forward pass for TxtEncoder.
        Args:
        - txt: List of text sequences (e.g., ["hello world", "this is an example"])
        Returns:
        - attention_value: Output of self-attention stack
        """
        # Tokenize text and convert to embeddings
        tokenized = self.tokenizer(txt, return_tensors="pt", padding=True, truncation=True).to(self.device)  # input_ids: (B, N_seq)
        input_ids = tokenized["input_ids"].to(self.device)  # Shape: (B, N_seq)

        # Convert token IDs to embeddings using nn.Embedding
        embeddings = self.embedding(input_ids)  # Shape: (B, N_seq, D_embed)

        # Positional encoding and self-attention
        positioned_txt = self.positionalEncoding1d(embeddings)  # Shape: (B, N_seq, D_embed)
        attention_value = self.selfAttention(positioned_txt)  # Shape: (B, N_seq, D_embed)
        return attention_value

class CrossEncoder(nn.Module):
    def __init__(self, 
                D_embed, N_heads, dropout=0.1, device='cpu'):
        super(CrossEncoder, self).__init__()
        self.device = device
        self.crossEncoder = CrossAttentionStack(D_embed, N_heads, dropout=dropout, device=device)  # Cross-attention block
        
    def forward(self, decoder_input, encoder_output):
        '''
        decoder Input shape : (B, N_seq_decoder, D_embed)
        encoder Input shape : (B, N_seq_encoder, D_embed)
        Returns:
        - out: (B, N_seq_decoder, D_embed)
        '''
        decoder_input = decoder_input.to(self.device)
        encoder_output = encoder_output.to(self.device)
        out = self.crossEncoder(decoder_input, encoder_output) # out: (B, N_seq_decoder, D_embed)
        return out 
    
class ImgDecoder(nn.Module):
    def __init__(self,
                 D_embed, h_dim, n_res_layers, res_h_dim, device='cpu'):
        super(ImgDecoder, self).__init__()
        self.device = device
        self.imgDecoder = VQVAE_Decoder(D_embed, h_dim, n_res_layers, res_h_dim).to(device)  # (B, N_seq_img, D_embed)
    
    def forward(self, z_q):
        z_q = z_q.to(self.device)
        # Reshape sequence to 2D grid for decoding
        N_seq_img = z_q.shape[1]
        W_flat = int(N_seq_img**0.5)  # Assume square spatial dimension
        H_flat = W_flat
        img_reconstructed = self.imgDecoder(z_q, W_flat, H_flat)  # (B, C, H, W)
        return img_reconstructed
    
class TxtDecoder(nn.Module):
    def __init__(self,
                 N_vocab: int, D_embed: int, N_heads: int, n_layers: int, device='cpu'):
        super(TxtDecoder, self).__init__()
        self.device = device
        self.txtDecoder = DecoderHead(N_vocab, D_embed, N_heads, n_layers).to(device)  # (B, N_seq, D_embed) -> (B, N_seq, N_vocab)
    
    def forward(self, x):
        """
        Forward pass for TxtDecoder
        Args:
        - x: Input tensor of shape (B, N_seq, D_embed)
        Returns:
        - output: Tensor of shape (B, N_seq, N_vocab)
        """
        x = x.to(self.device)
        return self.txtDecoder(x)

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
                             embedding_dim=D_embed, beta=beta, N_stacks=N_stacks, N_heads=N_heads, device=device).to(device)
    
    txt_encoder = TxtEncoder(pretrained_model_name="bert-base-uncased",
                             N_stacks=N_stacks, D_embed=D_embed, N_heads=N_heads, device=device).to(device)
    
    cross_encoder = CrossEncoder(D_embed=D_embed, N_heads=N_heads, device=device).to(device)
    
    img_decoder = ImgDecoder(embedding_dim=D_embed, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim, device=device).to(device)
    
    txt_decoder = TxtDecoder(N_vocab=N_vocab, D_embed=D_embed, N_heads=N_heads, n_layers=n_layers, device=device).to(device)

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
