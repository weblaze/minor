import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.utils as vutils
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.datasets import ImageDataset

# Paths
IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "abstract_art")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "image_vae.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "reconstructed_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
LATENT_DIM = 512
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BETA = 0.005  # Small KL weight as in original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
dataset = ImageDataset(IMAGE_PATH, train=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Loaded {len(dataset)} image files.")

# Model and optimizer
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
optimizer_vae = torch.optim.Adam(image_vae.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=20, gamma=0.5)

# VGG16 for style loss
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

vgg_preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

def style_loss(x, recon_x):
    layers = [2, 7, 12, 21, 30]  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
    loss = 0
    x_vgg = vgg_preprocess(x.detach().clone())
    recon_x_vgg = vgg_preprocess(recon_x.detach().clone())
    for i, layer in enumerate(vgg):
        x_vgg = layer(x_vgg)
        recon_x_vgg = layer(recon_x_vgg)
        if i in layers:
            gram_x = gram_matrix(x_vgg)
            gram_recon = gram_matrix(recon_x_vgg)
            loss += nn.functional.mse_loss(gram_x, gram_recon)
    return loss

def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    style_loss_val = style_loss(x, recon_x)
    mu_clamped = torch.clamp(mu, -10, 10)
    logvar_clamped = torch.clamp(logvar, -10, 10)
    kl_div = -0.5 * torch.mean(1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())
    return mse_loss + 20.0 * style_loss_val + beta * kl_div, mse_loss, style_loss_val, kl_div

# Training loop
image_vae.train()
for epoch in range(NUM_EPOCHS):
    total_loss, total_mse, total_style, total_kl = 0, 0, 0, 0
    
    # Warm-up phase for KL loss
    beta = 0.0 if epoch < 10 else min(BETA, BETA * (epoch - 10 + 1) / 10)
    # Warm-up phase for learning rate
    lr = min(LEARNING_RATE, 1e-5 + (LEARNING_RATE - 1e-5) * (epoch + 1) / 10) if epoch < 10 else LEARNING_RATE
    for param_group in optimizer_vae.param_groups:
        param_group['lr'] = lr

    for batch_idx, batch in enumerate(dataloader):
        images = batch.to(device)

        optimizer_vae.zero_grad()
        recon_images, mu, logvar = image_vae(images)
        
        # Debug prints for first batch
        if batch_idx == 0 and epoch % 5 == 0:
            print(f"Epoch {epoch+1}, Images min/max: {images.min().item():.4f}/{images.max().item():.4f}")
            print(f"Epoch {epoch+1}, Recon min/max: {recon_images.min().item():.4f}/{recon_images.max().item():.4f}")
            print(f"Epoch {epoch+1}, mu std: {mu.std().item():.4f}, logvar std: {logvar.std().item():.4f}")

        loss, mse_loss, style_loss_val, kl_div = vae_loss(recon_images, images, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(image_vae.parameters(), max_norm=1.0)
        optimizer_vae.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_style += style_loss_val.item()
        total_kl += kl_div.item()

        # Save images every 10 epochs
        if batch_idx == 0 and (epoch + 1) % 10 == 0:
            vutils.save_image(recon_images, os.path.join(OUTPUT_DIR, f"recon_epoch_{epoch+1}.png"), normalize=True)
            vutils.save_image(images, os.path.join(OUTPUT_DIR, f"orig_epoch_{epoch+1}.png"), normalize=True)

    scheduler_vae.step()

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_style = total_style / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, Style: {avg_style:.4f}, KL: {avg_kl:.4f}")

torch.save(image_vae.state_dict(), IMAGE_MODEL_PATH)
print(f"✅ Saved trained ImageVAE to {IMAGE_MODEL_PATH}")