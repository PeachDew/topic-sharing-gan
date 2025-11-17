import random
import torch
import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

    def generate(self, n=1):
        """Generate n adversarial outputs"""
        z = torch.randn(n, self.latent_dim)
        with torch.no_grad():
            out = self.forward(z)
        return z, out

def surrogate_loss(fake, real_batch):
    target = real_batch.mean(dim=0, keepdim=True).to(fake.device)
    return torch.mean((fake - target)**2)

class D_Trainer:
    def __init__(self):
        self.z = None   

def step_human_round(G, real_loader):
    # --- sample 3 real images ---
    real_imgs, _ = next(iter(real_loader))  # [B,1,28,28]
    real_imgs = real_imgs[:3]

    # --- generate 1 fake ---
    z, fake = G.generate()      # (1,1,28,28)
    fake = fake.detach()

    # --- random insert fake into 4 positions ---
    images = list(real_imgs)  
    fake_index = random.randint(0, 3)
    images.insert(fake_index, fake.squeeze(0))

    # =============================================
    # HUMAN SELECTS WHICH IMAGE THEY THINK IS FAKE
    # Replace this later with a real UI:
    # clicked_idx = st.button(...)
    # =============================================
    # For testing: simulate a human ~80% accurate
    clicked_idx = fake_index if random.random() < 1.0 else random.randint(0,3)

    # reward: 1 = fooled human, 0 = human spotted fake
    reward = 1 if clicked_idx != fake_index else 0

    return z, reward

def train_generator_step(G, opt_G, z, reward, real_batch, alpha=1.0):
    G.train()
    fake = G(z)

    # surrogate loss
    surr = surrogate_loss(fake, real_batch)

    # reward term (non-diff); moves generator toward fooling humans
    # reward = 1 → reduce loss; reward = 0 → no effect
    loss = surr - alpha * reward

    opt_G.zero_grad()
    loss.backward()
    opt_G.step()

    return loss.item(), surr.item()
