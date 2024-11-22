import torch
import torch.nn as nn
import logging
from torchvision.models import vit_b_16, ViT_B_16_Weights
from attention import AttentionBlock, AttentionStack  # 기존 모듈

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ViTEncoderLoader:
    """
    Utility to load pretrained ViT encoder weights into existing AttentionBlock and AttentionStack.
    """
    def __init__(self, pretrained_vit, num_layers):
        """
        Args:
        - pretrained_vit: Pretrained Vision Transformer model
        - num_layers: Number of encoder layers to use
        """
        self.vit = pretrained_vit
        self.encoder_layers = list(self.vit.encoder.layers.children())[:num_layers]
        logger.info(f"Loaded {len(self.encoder_layers)} encoder layers from pretrained ViT model.")

    def _split_qkv(self, in_proj_weight, in_proj_bias, D_embed):
        """
        Split Query, Key, Value weights and biases from combined QKV weights/biases.
        Args:
        - in_proj_weight: Combined QKV weight tensor of shape (3*D_embed, D_embed)
        - in_proj_bias: Combined QKV bias tensor of shape (3*D_embed)
        - D_embed: Embedding dimension
        Returns:
        - (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias)
        """
        logger.debug(f"Splitting QKV weights and biases with D_embed={D_embed}.")
        q_weight = in_proj_weight[:D_embed, :]
        k_weight = in_proj_weight[D_embed:2*D_embed, :]
        v_weight = in_proj_weight[2*D_embed:, :]

        q_bias = in_proj_bias[:D_embed]
        k_bias = in_proj_bias[D_embed:2*D_embed]
        v_bias = in_proj_bias[2*D_embed:]

        return (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias)

    def load_attention_block(self, block: AttentionBlock, layer_idx: int):
        """
        Load weights from a ViT encoder block into an AttentionBlock.
        Args:
        - block: Existing AttentionBlock instance
        - layer_idx: Index of the ViT encoder layer to load weights from
        """
        logger.info(f"Loading weights for AttentionBlock at layer {layer_idx}.")
        vit_layer = self.encoder_layers[layer_idx]

        try:
            # Extract Multi-Head Attention weights
            in_proj_weight = vit_layer.self_attention.in_proj_weight  # Shape: (3*D_embed, D_embed)
            in_proj_bias = vit_layer.self_attention.in_proj_bias      # Shape: (3*D_embed)
            out_proj_weight = vit_layer.self_attention.out_proj.weight  # Shape: (D_embed, D_embed)
            out_proj_bias = vit_layer.self_attention.out_proj.bias      # Shape: (D_embed)

            D_embed = block.attention.D_embed

            # Split QKV weights and biases
            (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias) = self._split_qkv(in_proj_weight, in_proj_bias, D_embed)

            # Map QKV weights and biases to MultiHeadAttention
            block.attention.w_q.weight.data = q_weight
            block.attention.w_q.bias.data = q_bias
            block.attention.w_k.weight.data = k_weight
            block.attention.w_k.bias.data = k_bias
            block.attention.w_v.weight.data = v_weight
            block.attention.w_v.bias.data = v_bias

            # Map output projection weights and biases
            block.attention.w_o.weight.data = out_proj_weight
            block.attention.w_o.bias.data = out_proj_bias

            # Extract Feed-Forward Network (FFNN) weights
            block.FFNN.fc1.weight.data = vit_layer.mlp[0].weight.data
            block.FFNN.fc1.bias.data = vit_layer.mlp[0].bias.data
            block.FFNN.fc2.weight.data = vit_layer.mlp[3].weight.data
            block.FFNN.fc2.bias.data = vit_layer.mlp[3].bias.data

            logger.info(f"Successfully loaded weights for layer {layer_idx}.")
        except Exception as e:
            logger.error(f"Failed to load weights for layer {layer_idx}: {e}")


    def load_attention_stack(self, stack: AttentionStack):
        """
        Load weights into an AttentionStack.
        Args:
        - stack: Existing AttentionStack instance
        """
        logger.info("Loading weights into AttentionStack.")
        for i, block in enumerate(stack.stack):
            self.load_attention_block(block, i)
        logger.info("Finished loading weights into AttentionStack.")

# Example usage
if __name__ == "__main__":
    # Step 1: Load pretrained ViT with weights
    weights = ViT_B_16_Weights.DEFAULT
    pretrained_vit = vit_b_16(weights=weights)
    logger.info("Pretrained ViT model loaded successfully.")

    # Step 2: Define parameters
    D_embed = 768  # Embedding dimension
    N_heads = 12   # Number of attention heads
    num_layers = 3  # Number of layers to use

    # Step 3: Initialize AttentionStack
    logger.info(f"Initializing AttentionStack with {num_layers} layers, D_embed={D_embed}, N_heads={N_heads}.")
    attention_stack = AttentionStack(n=num_layers, D_embed=D_embed, N_heads=N_heads)

    # Step 4: Load pretrained weights into AttentionStack
    vit_loader = ViTEncoderLoader(pretrained_vit, num_layers)
    vit_loader.load_attention_stack(attention_stack)

    # Step 5: Test with dummy input
    B, N_seq, D_embed = 1, 197, 768  # Batch size, Sequence length (197 = 196 patches + 1 CLS token), Embedding dimension
    dummy_input = torch.randn(B, N_seq, D_embed)
    logger.info(f"Running forward pass on AttentionStack with input shape: {dummy_input.shape}")

    output = attention_stack(dummy_input)
    logger.info(f"Output shape from AttentionStack: {output.shape}")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")  # Expected: (B, N_seq, D_embed)
