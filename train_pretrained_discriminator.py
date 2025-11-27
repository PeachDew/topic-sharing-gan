import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.gan import Discriminator, Generator

MODEL_PATH = "models/pretrained_dicriminator.pth"

def train_discriminator_on_real_data(epochs=10, batch_size=64, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load MNIST 5s
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    # Filter only 5s
    indices = [i for i, label in enumerate(mnist_data.targets) if label == 5]
    mnist_5s = Subset(mnist_data, indices)
    
    print(f"Training on {len(mnist_5s)} real 5s")
    
    dataloader = DataLoader(mnist_5s, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)  # For generating fake samples
    
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    
    # Training loop
    d_losses = []
    
    for epoch in range(epochs):
        epoch_d_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real_images, _ in progress_bar:
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_validity = discriminator(real_images)
            d_real_loss = criterion(real_validity, real_labels)
            
            # Fake images from generator
            z = torch.randn(batch_size_actual, generator.latent_dim).to(device)
            fake_images = generator(z).detach()
            fake_validity = discriminator(fake_images)
            d_fake_loss = criterion(fake_validity, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            z = torch.randn(batch_size_actual, generator.latent_dim).to(device)
            fake_images = generator(z)  
            fake_validity = discriminator(fake_images)
            g_loss = criterion(fake_validity, real_labels)  
            g_loss.backward()
            g_optimizer.step()
            
            epoch_d_loss += d_loss.item()
            progress_bar.set_postfix({'d_loss': d_loss.item()})
        
        avg_d_loss = epoch_d_loss / len(dataloader)
        d_losses.append(avg_d_loss)
        print(f"Epoch {epoch+1}/{epochs} - Avg D Loss: {avg_d_loss:.4f}")
        
        # Test discriminator
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                # Test on real image
                test_real, _ = mnist_5s[0]
                test_real = test_real.unsqueeze(0).to(device)
                real_score = discriminator(test_real).item()
                
                # Test on fake image
                test_z = torch.randn(1, generator.latent_dim).to(device)
                test_fake = generator(test_z)
                fake_score = discriminator(test_fake).item()
                
                print(f"  Real score: {real_score:.4f}, Fake score: {fake_score:.4f}")
     
    # Save the trained discriminator
    torch.save({
        'model_state_dict': discriminator.state_dict(),
        'epoch': epochs,
        'loss': d_losses[-1],
    }, MODEL_PATH)
    
    print(f"\n✅ Saved pretrained discriminator to: {MODEL_PATH}")
    
    return discriminator

def test_discriminator(discriminator_path=MODEL_PATH):
    """
    Test the pretrained discriminator on some examples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load discriminator
    discriminator = Discriminator().to(device)
    checkpoint = torch.load(discriminator_path, map_location=device)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator.eval()
    
    # Load some real MNIST 5s
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    indices = [i for i, label in enumerate(mnist_data.targets) if label == 5]
    mnist_5s = Subset(mnist_data, indices)
    
    # Test on real images
    print("\nTesting on real MNIST 5s:")
    with torch.no_grad():
        for i in range(5):
            img, _ = mnist_5s[i]
            img = img.unsqueeze(0).to(device)
            score = discriminator(img).item()
            print(f"  Real image {i+1}: {score:.4f}")
    
    # Test on untrained generator
    print("\nTesting on untrained generator:")
    generator = Generator().to(device)
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, generator.latent_dim).to(device)
            fake = generator(z)
            score = discriminator(fake).item()
            print(f"  Fake image {i+1}: {score:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Training Pretrained Discriminator for MNIST 5s")
    print("=" * 60)
    
    # Train discriminator
    discriminator = train_discriminator_on_real_data(epochs=11, batch_size=64)
    
    test_discriminator()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
