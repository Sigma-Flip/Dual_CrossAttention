import torch
import torch.nn as nn
import torch.nn.functional as F
from vqvae.vqvae import VQVAE_Decoder
from transformer import AttentionStack, CrossAttentionStack
from transformer import PositionalEncoding
from transformer import DecoderHead
from transformers import AutoTokenizer, ViTModel, ViTImageProcessor

# Initialize ViT model and image processor
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')


class TxtEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str, N_stacks: int, D_embed: int, N_heads: int):
        super(TxtEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)  # Pre-trained tokenizer
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, D_embed)  # Embedding layer to convert input_ids to D_embed
        self.positionalEncoding1d = PositionalEncoding(D_embed)  # 1D Positional Encoding
        self.selfAttention = AttentionStack(N_stacks, D_embed, N_heads)  # Attention Stack

    def forward(self, txt):
        tokenized = self.tokenizer(txt, return_tensors="pt", padding=True, truncation=True)  # input_ids: (B, N_seq)
        input_ids = tokenized["input_ids"].to(device)  # 텐서를 디바이스로 옮깁니다.

        # Convert token IDs to embeddings using nn.Embedding
        embeddings = self.embedding(input_ids)  # Shape: (B, N_seq, D_embed)

        # Positional encoding and self-attention
        positioned_txt = self.positionalEncoding1d(embeddings)  # Shape: (B, N_seq, D_embed)
        attention_value = self.selfAttention(positioned_txt)  # Shape: (B, N_seq, D_embed)
        return attention_value


class CrossEncoder(nn.Module):
    def __init__(self, D_embed, N_heads, dropout=0.1):
        super(CrossEncoder, self).__init__()
        self.crossEncoder = CrossAttentionStack(D_embed, N_heads, dropout=0.1)

    def forward(self, decoder_input, encoder_output):
        out = self.crossEncoder(decoder_input, encoder_output)  # out: (B, N_seq_decoder, D_embed)
        return out


class ImgDecoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, n_res_layers, res_h_dim):
        super(ImgDecoder, self).__init__()
        self.imgDecoder = VQVAE_Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, z_q):
        N_seq_img = z_q.shape[1]
        W_flat = int(N_seq_img**0.5)  # Assume square spatial dimension
        H_flat = W_flat
        img_reconstructed = self.imgDecoder(z_q, W_flat, H_flat)  # (B, C, H, W)
        return img_reconstructed


class TxtDecoder(nn.Module):
    def __init__(self, N_vocab: int, D_embed: int, N_heads: int, n_layers: int):
        super(TxtDecoder, self).__init__()
        self.txtDecoder = DecoderHead(N_vocab, D_embed, N_heads, n_layers)

    def forward(self, x):
        return self.txtDecoder(x)


# Main script
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters
    B, C, W, H = 4, 3, 224, 224  # Image shape: Batch size, Channels, Width, Height (adapted to ViT input size)
    N_vocab, D_embed, N_heads, N_stacks, n_layers = 30522, 768, 8, 3, 2  # Text-related parameters
    h_dim, res_h_dim, n_res_layers, n_embeddings, beta = 128, 64, 3, 512, 0.25  # VQ-VAE parameters

    # Example input data
    img_data = torch.randn(B, C, W, H).to(device)  # Random image tensor
    txt_data = ["Hello world", "This is a test", "Transformer example", "PyTorch integration"]

    # Normalize image data to range [0, 1]
    img_data = torch.clamp(img_data, 0, 1)

    # Move pretrained ViT model to device
    vit_img_encoder.to(device)

    # Forward pass through ViT Image Encoder
    vit_features = image_processor(images=img_data.permute(0, 2, 3, 1).cpu().numpy(), return_tensors="pt", do_rescale=False)  # Extract features using ViT image processor
    vit_features = {k: v.to(device) for k, v in vit_features.items()}  # Move features to device
    img_attention_value = vit_img_encoder(**vit_features).last_hidden_state  # Output: (B, N_seq_img, D_embed)

    # Initialize other models
    txt_encoder = TxtEncoder(pretrained_model_name="bert-base-uncased",
                             N_stacks=N_stacks, D_embed=D_embed, N_heads=N_heads).to(device)
    cross_encoder = CrossEncoder(D_embed=D_embed, N_heads=N_heads).to(device)
    img_decoder = ImgDecoder(embedding_dim=D_embed, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim).to(device)
    txt_decoder = TxtDecoder(N_vocab=N_vocab, D_embed=D_embed, N_heads=N_heads, n_layers=n_layers).to(device)

    # Forward pass for text
    txt_attention_value = txt_encoder(txt_data)  # txt_data : (B, N_seq)

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
