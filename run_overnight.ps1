# Overnight training script for Abstraction
# Runs Stage 1 (VAE Finetuning) and Stage 2 (Latent Diffusion) sequentially

$env:WANDB_MODE="offline"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 1: Finetuning VAE decoder with latent dropout (Approx 1.5h)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

python approaches/01_conditioned_autoencoder/train.py --stage finetune

if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 1 (VAE Finetuning) failed with exit code $LASTEXITCODE. Aborting Stage 2." -ForegroundColor Red
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 1 COMPLETE. Checkpoint saved to tmodels/01_conditioned_autoencoder/image_vae_conditioned.pth" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 2: Training Latent Diffusion UNet (Approx 2h)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

python approaches/02_latent_diffusion_clap/train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 2 (Latent Diffusion) failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 2 COMPLETE. Checkpoint saved to tmodels/02_latent_diffusion_clap/unet_own.pth" -ForegroundColor Green
Write-Host "ALL TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
