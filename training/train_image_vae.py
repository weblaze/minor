# D:\musicc\minor\training\train_image_vae.py
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "abstract_art")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "image_vae.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "reconstructed_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.datasets import ImageDataset
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.transforms as transforms

# Enable anomaly detection to catch inplace operations
torch.autograd.set_detect_anomaly(True)



LATENT_DIM = 256
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BETA = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = ImageDataset(IMAGE_PATH, train=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Discriminator for adversarial loss
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Conv2d(512, 1, 8, stride=1, padding=0)  # 8x8 -> 1x1
        self.feature_layers = [0, 2, 4]  # Layers for feature matching

    def forward(self, x):
        features = []
        h = x
        for i, layer in enumerate(self.model):
            h = layer(h)
            if i in self.feature_layers:
                features.append(h)
        out = self.classifier(h).view(-1, 1)
        return torch.sigmoid(out), features

image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

optimizer_vae = torch.optim.Adam(image_vae.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=20, gamma=0.5)
scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=20, gamma=0.5)

# VGG16 for style loss
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# VGG16 preprocessing
vgg_preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

def style_loss(x, recon_x):
    layers = [2, 7, 12, 21, 30]  # VGG16 layers: conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
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

def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_f, fake_f in zip(real_features, fake_features):
        loss += nn.functional.mse_loss(real_f, fake_f)
    return loss / len(real_features)

def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    style_loss_val = style_loss(x, recon_x)
    mu_clamped = torch.clamp(mu, -10, 10, out=None)
    logvar_clamped = torch.clamp(logvar, -10, 10, out=None)
    kl_div = -0.5 * torch.mean(1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())
    return mse_loss + 20.0 * style_loss_val + beta * kl_div, mse_loss, style_loss_val, kl_div

def adversarial_loss(recon_x, real_x):
    real_pred, real_features = discriminator(real_x)
    fake_pred, fake_features = discriminator(recon_x)
    # Label smoothing
    real_loss = nn.functional.binary_cross_entropy(real_pred, torch.ones_like(real_pred) * 0.9)
    fake_loss = nn.functional.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss) / 2, fake_pred, real_features, fake_features

image_vae.train()
discriminator.train()
for epoch in range(NUM_EPOCHS):
    total_loss, total_mse, total_style, total_kl, total_adv, total_fm = 0, 0, 0, 0, 0, 0
    # Warm-up phase for KL loss
    beta = 0.0 if epoch < 10 else min(BETA, BETA * (epoch - 10 + 1) / 10)
    # Warm-up phase for learning rate
    lr = min(LEARNING_RATE, 1e-5 + (LEARNING_RATE - 1e-5) * (epoch + 1) / 10) if epoch < 10 else LEARNING_RATE
    for param_group in optimizer_vae.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_disc.param_groups:
        param_group['lr'] = lr

    for batch_idx, batch in enumerate(dataloader):
        images = batch.to(device)

        # Train Discriminator (only after 50 epochs)
        if epoch >= 50 and batch_idx % 20 == 0:
            optimizer_disc.zero_grad()
            recon_images, mu, logvar = image_vae(images)
            disc_loss, _, _, _ = adversarial_loss(recon_images, images)
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_disc.step()

        # Train VAE
        optimizer_vae.zero_grad()
        recon_images, mu, logvar = image_vae(images)
        vae_loss_val, mse_loss, style_loss_val, kl_div = vae_loss(recon_images, images, mu, logvar, beta=beta)
        adv_loss, fm_loss = 0.0, 0.0
        if epoch >= 50:  # Introduce adversarial loss after 50 epochs
            _, fake_pred, real_features, fake_features = adversarial_loss(recon_images, images)
            adv_loss = nn.functional.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred) * 0.9)
            fm_loss = feature_matching_loss(real_features, fake_features)
        total_loss_val = vae_loss_val + 1.0 * adv_loss + 2.0 * fm_loss
        total_loss_val.backward()

        # Gradient clipping for encoder and decoder parameters
        encoder_params = [p for n, p in image_vae.named_parameters() if 'encoder_layers' in n or 'fc_mu' in n or 'fc_logvar' in n]
        decoder_params = [p for n, p in image_vae.named_parameters() if 'decoder_input' in n or 'decoder_layers' in n]
        torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=1.0)
        optimizer_vae.step()

        total_loss += total_loss_val.item()
        total_mse += mse_loss.item()
        total_style += style_loss_val.item()
        total_kl += kl_div.item()
        total_adv += adv_loss
        total_fm += fm_loss

        if batch_idx == 0 and (epoch + 1) % 10 == 0:
            vutils.save_image(recon_images, os.path.join(OUTPUT_DIR, f"recon_epoch_{epoch+1}.png"), normalize=True)
            vutils.save_image(images, os.path.join(OUTPUT_DIR, f"orig_epoch_{epoch+1}.png"), normalize=True)

    scheduler_vae.step()
    scheduler_disc.step()

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_style = total_style / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_adv = total_adv / len(dataloader)
    avg_fm = total_fm / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, Style: {avg_style:.4f}, KL: {avg_kl:.4f}, Adv: {avg_adv:.4f}, FM: {avg_fm:.4f}")

torch.save(image_vae.state_dict(), IMAGE_MODEL_PATH)
print(f"✅ Saved trained ImageVAE to {IMAGE_MODEL_PATH}")