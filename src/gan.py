import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Generator(nn.Module):
    def __init__(self, latent_dim=64, img_size=28, channels=1):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Calculate initial spatial size
        self.init_size = img_size // 4  # 7 for 28x28 images
        
        # Project and reshape
        self.fc = nn.Linear(latent_dim, 64 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            
            # Upsample to 14x14
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        # Project and reshape
        out = self.fc(z)
        out = out.view(out.size(0), 64, self.init_size, self.init_size)
        # Generate image
        img = self.conv_blocks(out)
        return img

    def generate(self, n=1):
        """Generate n adversarial outputs"""
        z = torch.randn(n, self.latent_dim)
        with torch.no_grad():
            out = self.forward(z)
        return z, out

class Discriminator(nn.Module):
    def __init__(self, img_size=28, channels=1):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        
        # CNN feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),  # 28x28
            nn.ReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(0.2),
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        # CNN features: (B, 128, 7, 7)
        features = self.conv(img)
        
        # Reshape for attention: (B, 49, 128)
        B, C, H, W = features.shape
        features = features.view(B, C, H*W).permute(0, 2, 1)
        
        # Self-attention
        attn_out, _ = self.attention(features, features, features)
        
        # Global average pooling: (B, 128)
        pooled = attn_out.mean(dim=1)
        
        # Classification
        validity = self.classifier(pooled)
        return validity

def get_loss(pred: float): # 0 <= pred <= 1
    return 1-pred

def generate_image(generator, latent=None, scale=10):
    """Generate an image, optionally from specific latent vector"""
    if latent is None:
        latent = torch.randn(1, generator.latent_dim)
    
    with torch.no_grad():
        out = generator(latent)
    
    img = out[0, 0].cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((28 * scale, 28 * scale), Image.NEAREST)
    return pil_img

def array_to_pil(arr, scale=1):
    img = arr[0,0].cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((28 * scale, 28 * scale), Image.NEAREST)
    return pil_img

def get_random_real_image(mnist_data, scale=10):
    """Return a random MNIST real image as a scaled-up PIL image."""
    idx = random.randint(0, len(mnist_data) - 1)
    img, _ = mnist_data[idx]           # Tensor shape [1, 28, 28]

    # Convert from [-1, 1] → [0, 255]
    img = img[0].numpy()               # (28,28)
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((28 * scale, 28 * scale), Image.NEAREST)
    return pil_img

def get_real_image_batch(mnist_data, batch_size=32):
    """Get a batch of real images as tensors for training"""
    indices = random.sample(range(len(mnist_data)), batch_size)
    images = []
    
    for idx in indices:
        img, _ = mnist_data[idx]
        images.append(img)
    
    return torch.stack(images)

def train_gan_step(generator, discriminator, real_images, human_feedback=None, 
                    g_optimizer=None, d_optimizer=None, num_d_steps=1, human_feedback_boost=1.0):
    """
    Perform one GAN training step.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        real_images: Batch of real images from MNIST (tensor)
        human_feedback: Dict with {'latent': z, 'score': human_score} or None
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        num_d_steps: Number of discriminator updates per generator update
    
    Returns:
        dict with losses
    """
    criterion = nn.BCELoss()
    batch_size = real_images.size(0)
    
    # Labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    # Train Discriminator
    d_loss_total = 0
    for _ in range(num_d_steps):
        d_optimizer.zero_grad()
        
        # Train on real images
        real_validity = discriminator(real_images)
        d_real_loss = criterion(real_validity, real_labels)
        
        # Train on fake images
        z = torch.randn(batch_size, generator.latent_dim)
        fake_images = generator(z)
        fake_validity = discriminator(fake_images.detach())
        d_fake_loss = criterion(fake_validity, fake_labels)
        
        # If we have human feedback, incorporate it
        if human_feedback is not None:
            human_z = human_feedback['latent']
            human_score = human_feedback['score']  # 0-1, where 1 is "looks real"
            
            human_image = generator(human_z)
            human_validity = discriminator(human_image.detach())
            
            # Human score is the "true" label for this image
            human_label = torch.tensor([[human_score]])
            human_loss = criterion(human_validity, human_label)
            
            # Weight human feedback more heavily
            d_loss = d_real_loss + d_fake_loss + human_feedback_boost * human_loss
        else:
            d_loss = d_real_loss + d_fake_loss
        
        d_loss.backward()
        d_optimizer.step()
        d_loss_total += d_loss.item()
    
    # Train Generator
    g_optimizer.zero_grad()
    
    # Generate fake images
    z = torch.randn(batch_size, generator.latent_dim)
    fake_images = generator(z)
    
    # Generator wants discriminator to think these are real
    fake_validity = discriminator(fake_images)
    g_loss = criterion(fake_validity, real_labels)
    
    g_loss.backward()
    g_optimizer.step()
    
    return {
        'd_loss': d_loss_total / num_d_steps,
        'g_loss': g_loss.item(),
        'human_feedback_used': human_feedback is not None
    }
