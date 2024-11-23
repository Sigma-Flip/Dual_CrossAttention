import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import random
from transformers import BertTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import zipfile
import requests
from pathlib import Path
from PIL import Image

from modules import *  # Assuming custom modules are imported here
from transformer import ViTEncoderLoader  # Assuming custom transformer module

# Parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 3
CHECKPOINT_DIR = "/home/dongwoo42/new_dongwoo/results"
FLICKR8K_ROOT = '/home/dongwoo42/data/flickr8k/Flicker8k_Dataset'
FLICKR8K_ANNOTATIONS = '/home/dongwoo42/data/flickr8k/Flickr8k.token.txt'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
OUTPUT_DIR = './output'
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

# Model Hyperparameters
D_embed = 64
H = 64
W = 64
h_dim = 128
res_h_dim = 64
n_res_layers = 3
n_embeddings = 512
beta = 0.25
N_stacks = 3
N_heads = 4
n_layers = 2
N_vocab = 30522

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Download Flickr8k dataset if not exists
data_dir = Path(FLICKR8K_ROOT)
annotations_file = Path(FLICKR8K_ANNOTATIONS)

# Custom Flickr8k Dataset to handle '#n' in captions
class CustomFlickr8k(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.captions = {}

        with open(ann_file, 'r') as f:
            for line in f:
                img_id, caption = line.split('\t')
                img_id = img_id.split('#')[0]  # Remove '#n' from image ID
                if img_id not in self.captions:
                    self.images.append(img_id)
                    self.captions[img_id] = []
                self.captions[img_id].append(caption.strip())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.root, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        captions = self.captions[img_id]
        return image, random.choice(captions)

# DDP Setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger.info(f"Process group initialized for rank {rank}")

def cleanup():
    dist.destroy_process_group()
    logger.info("Process group destroyed")

# Main training function
def main(rank, world_size):
    setup(rank, world_size)

    # Load dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Handle dataset splits for training, validation, and test
    full_dataset = CustomFlickr8k(root=FLICKR8K_ROOT, ann_file=FLICKR8K_ANNOTATIONS, transform=transform)

    # Verify dataset loading
    if len(full_dataset) == 0:
        logger.error("Dataset could not be loaded properly. Please check the dataset path and files.")
        cleanup()
        return

    dataset_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    test_size = int(TEST_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # Debugging: Log dataset sizes
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=2)

    # Initialize models
    weights = ViT_B_16_Weights.DEFAULT
    pretrained_vit = vit_b_16(weights=weights).to(rank)
    pretrained_attention_stack = AttentionStack(n=N_stacks, D_embed=D_embed, N_heads=N_heads, device=rank).to(rank)
    vit_loader = ViTEncoderLoader(pretrained_vit, num_layers=N_stacks)
    vit_loader.load_attention_stack(pretrained_attention_stack)

    img_encoder = ImgEncoder(D_embed=D_embed, H=H, W=W, h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers,
                             n_embeddings=D_embed, beta=beta, N_stacks=N_stacks, N_heads=N_heads, device=rank).to(rank)
    img_encoder.selfAttention = pretrained_attention_stack

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "epoch_400_model.pth"), map_location=lambda storage, loc: storage.cuda(rank))
    vqvae_encoder = VQVAE_Encoder(D_embed=D_embed,in_dim=3, h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers, n_embeddings=n_embeddings, beta=beta).to(rank)
    vqvae_decoder = VQVAE_Decoder(D_embed=D_embed, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim).to(rank)

    # Load state_dict with strict=False to skip mismatched layers
    vqvae_encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
    vqvae_decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)

    img_encoder.vqvaeEncoder = vqvae_encoder
    img_decoder = ImgDecoder(D_embed=D_embed, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim, device=rank).to(rank)
    img_decoder.imgDecoder = vqvae_decoder

    txt_encoder = TxtEncoder(pretrained_model_name="bert-base-uncased", N_stacks=N_stacks, D_embed=D_embed, N_heads=N_heads, device=rank).to(rank)
    cross_encoder = CrossEncoder(D_embed=D_embed, N_heads=N_heads, device=rank).to(rank)
    txt_decoder = TxtDecoder(N_vocab=N_vocab, D_embed=D_embed, N_heads=N_heads, n_layers=n_layers, device=rank).to(rank)

    # Freeze parameters of ViT and VQVAE encoder models
    for param in pretrained_vit.parameters():
        param.requires_grad = False
    for param in vqvae_encoder.parameters():
        param.requires_grad = False

    # Wrap models with DDP
    img_encoder = DDP(img_encoder, device_ids=[rank])
    txt_encoder = DDP(txt_encoder, device_ids=[rank])
    cross_encoder = DDP(cross_encoder, device_ids=[rank])
    img_decoder = DDP(img_decoder, device_ids=[rank])
    txt_decoder = DDP(txt_decoder, device_ids=[rank])

    # Optimizers
    optimizer = optim.Adam(list(img_encoder.parameters()) + list(txt_encoder.parameters()) +
                           list(cross_encoder.parameters()) + list(img_decoder.parameters()) +
                           list(txt_decoder.parameters()), lr=LEARNING_RATE)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Starting epoch {epoch}")
        train_sampler.set_epoch(epoch)
        img_encoder.train()
        txt_encoder.train()
        cross_encoder.train()
        img_decoder.train()
        txt_decoder.train()

        epoch_loss = 0.0

        if len(train_loader) == 0:
            logger.warning("Train loader is empty. No data available for training.")

        with tqdm(train_loader, unit="batch", disable=(rank != 0)) as tepoch:
            tepoch.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            for batch_idx, (images, captions) in enumerate(tepoch):
                print(f"image shape : {images.shape}")

                images = images.to(rank)
                logger.debug(f"Batch {batch_idx}: Image shape: {images.shape}")  # Debugging image shape

                # Tokenize captions
                target_captions = [torch.tensor(tokenizer.encode(caption, add_special_tokens=False)).cuda(rank) for caption in captions]
                target_captions_padded = torch.nn.utils.rnn.pad_sequence(target_captions, batch_first=True, padding_value=tokenizer.pad_token_id).to(rank)
                logger.debug(f"Batch {batch_idx}: Target captions padded shape: {target_captions_padded.shape}")  # Debugging caption shape

                # Text generation from image
                optimizer.zero_grad()
                img_attention_value, img_embedding_loss, img_perplexity, z_e, _, _ = img_encoder(images)
                logger.debug(f"Batch {batch_idx}: img_attention_value shape: {img_attention_value.shape}")  # Debugging attention value shape
                sos_token = tokenizer.cls_token_id
                input_captions = torch.full((len(captions), 1), sos_token, dtype=torch.long).cuda(rank)
                txt_attention_value = txt_encoder(input_captions)
                logger.debug(f"Batch {batch_idx}: txt_attention_value shape: {txt_attention_value.shape}")  # Debugging text attention value shape
                img2txt_encoded_txt = cross_encoder(decoder_input=txt_attention_value, encoder_output=img_attention_value)
                generated_captions = txt_decoder(img2txt_encoded_txt)
                logger.debug(f"Batch {batch_idx}: Generated captions shape: {generated_captions.shape}")  # Debugging generated captions shape
                caption_loss = nn.CrossEntropyLoss()(generated_captions.view(-1, generated_captions.size(-1)), target_captions_padded.view(-1))
                caption_loss.backward()
                optimizer.step()

                # Image generation from noisy image
                optimizer.zero_grad()
                noise = torch.randn_like(images) * 0.1
                noisy_images = images + noise
                img_attention_value, img_embedding_loss, img_perplexity, z_e, _, _ = img_encoder(noisy_images)
                txt_attention_value = txt_encoder(target_captions_padded)
                txt2img_encoded_img = cross_encoder(decoder_input=img_attention_value, encoder_output=txt_attention_value)
                reconstructed_images = img_decoder(txt2img_encoded_img)
                logger.debug(f"Batch {batch_idx}: Reconstructed images shape: {reconstructed_images.shape}")  # Debugging reconstructed image shape
                image_loss = nn.functional.mse_loss(reconstructed_images, images)
                image_loss.backward()
                optimizer.step()

                # Accumulate loss for logging
                total_loss = caption_loss.item() + image_loss.item()
                epoch_loss += total_loss

                # Update tqdm progress bar with loss values
                tepoch.set_postfix({"Total Loss": total_loss, "Caption Loss": caption_loss.item(), "Image Loss": image_loss.item()})

        if len(train_loader) > 0:
            logger.info(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")
        else:
            logger.info(f"Epoch [{epoch}/{EPOCHS}], No training data available")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

logger.info("Training Finished")
