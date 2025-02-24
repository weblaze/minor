import torch
import torch.nn as nn
import torch.optim as optim
from training.data_loader_audio_img import get_dataloader
from models.autoencoder.model import AudioToImageAutoencoder
import os
from pathlib import Path

def train():
    # Set paths
    audio_features_path = "data/normalized_features.npy"
    image_folder = "datasets/abstract_art"
    save_model_path = "models/autoencoder/audio_to_image_autoencoder.pth"

    # Create save directory
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataloader = get_dataloader(audio_features_path, image_folder, batch_size=8)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Initializing model...")
    model = AudioToImageAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # Training Loop
    print("Starting training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (audio, image) in enumerate(dataloader):
            audio, image = audio.to(device), image.to(device)
            optimizer.zero_grad()
            reconstructed_image = model(audio)
            loss = criterion(reconstructed_image, image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save the trained model
    print(f"Saving model to {save_model_path}")
    torch.save(model.state_dict(), save_model_path)
    print("Training completed!")

if __name__ == "__main__":
    train() 