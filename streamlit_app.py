import streamlit as st
from torchvision import datasets, transforms
from torch import optim

import sys
import os
import torch
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.gan import (
    Generator, 
    Discriminator,
    generate_image, 
    get_random_real_image,
    train_gan_step,
    get_real_image_batch
)
from pymongo import MongoClient

client = MongoClient(st.secrets.uri)

st.set_page_config(page_title="GAN", page_icon=":woozy:", layout=None, initial_sidebar_state=None, menu_items=None)
st.title("You are in the GAN")

if "mnist_data" not in st.session_state:
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
    
    indices = [i for i, label in enumerate(mnist_data.targets) if label == 5]
    st.session_state.mnist_data = torch.utils.data.Subset(mnist_data, indices)

WARMUP_STEPS = 0
CORRECT_IMAGES_PROB = 0.0

if "generator" not in st.session_state:
    st.session_state.generator = Generator()
if "discriminator" not in st.session_state:
    st.session_state.discriminator = Discriminator()
if "g_optimizer" not in st.session_state:
    st.session_state.g_optimizer = optim.Adam(
        st.session_state.generator.parameters(), 
        lr=0.0002, 
        betas=(0.5, 0.999)
    )
if "d_optimizer" not in st.session_state:
    st.session_state.d_optimizer = optim.Adam(
        st.session_state.discriminator.parameters(), 
        lr=0.0002, 
        betas=(0.5, 0.999)
    )

if "training_steps" not in st.session_state:
    st.session_state.training_steps = 0
if "last_losses" not in st.session_state:
    st.session_state.last_losses = {"d_loss": 0, "g_loss": 0}

st.badge(f"{int(st.session_state.training_steps)} Training Steps", width="stretch", icon=":material/star_shine:")

def train_with_ml_discriminator(num_steps=199):
    """
    Train the GAN using ML discriminator only (no human feedback).
    This happens between human guesses.
    """
    my_bar = st.progress(0, text="Training...")
    for i in range(num_steps):
        # Get a batch of real images
        real_batch = get_real_image_batch(st.session_state.mnist_data, batch_size=32)
        
        # Train GAN (no human feedback)
        losses = train_gan_step(
            generator=st.session_state.generator,
            discriminator=st.session_state.discriminator,
            real_images=real_batch,
            human_feedback=None,  # No human feedback in these steps
            g_optimizer=st.session_state.g_optimizer,
            d_optimizer=st.session_state.d_optimizer,
            num_d_steps=1  # 2 discriminator update per generator update
        )
        
        st.session_state.last_losses = losses
        my_bar.progress((i + 1)/num_steps)
    my_bar.empty()
    
    st.session_state.training_steps += num_steps

if st.session_state.training_steps == 0 and "current_image" not in st.session_state:
    with st.spinner("Warming up the generator..."):
        if WARMUP_STEPS > 0:
            train_with_ml_discriminator(WARMUP_STEPS)
            # Generate first image and store its latent vector
        latent = torch.randn(1, st.session_state.generator.latent_dim)
        st.session_state.current_latent = latent
        st.session_state.current_image = {
            "image": generate_image(st.session_state.generator, latent), 
            "fake": 1
        }


def next_image():
    """Generate and show next image, calculate points for previous guess"""
    is_fake = st.session_state.current_image["fake"]
    # if is_fake:
    #     st.toast("Fake!",icon="🤭")
    # else:
    #     st.toast("realsies")


    human_score = st.session_state.slider_value / 100.0
    if is_fake == 1 and st.session_state.current_latent is not None:
        # Get a batch of real images for this training step
        real_batch = get_real_image_batch(st.session_state.mnist_data, batch_size=32)
        
        # Create human feedback dict
        human_feedback = {
            'latent': st.session_state.current_latent,
            'score': human_score  # Human's assessment (0 = fake, 1 = real)
        }
        
        # Train with human feedback
        losses = train_gan_step(
            generator=st.session_state.generator,
            discriminator=st.session_state.discriminator,
            real_images=real_batch,
            human_feedback=human_feedback,  # Include human feedback!
            g_optimizer=st.session_state.g_optimizer,
            d_optimizer=st.session_state.d_optimizer,
            num_d_steps=1,
            human_feedback_boost=1.5
        )
        
        st.session_state.last_losses = losses
        st.session_state.training_steps += 1
    
        with st.spinner(f"Training GAN with ML discriminator ({st.session_state.num_ml_steps} steps)..."):
            train_with_ml_discriminator(num_steps=st.session_state.num_ml_steps)
    
    # 4. Generate next image
    if random.random() < CORRECT_IMAGES_PROB:
        # Show a real image
        st.session_state.current_latent = None
        st.session_state.current_image = {
            "image": get_random_real_image(st.session_state.mnist_data), 
            "fake": 0
        }
    else:
        # Generate a fake image and store its latent vector
        latent = torch.randn(1, st.session_state.generator.latent_dim)
        st.session_state.current_latent = latent
        st.session_state.current_image = {
            "image": generate_image(st.session_state.generator, latent), 
            "fake": 1
        }

c1,c2 = st.columns([1,2], vertical_alignment="center")
with c1:
    st.image(st.session_state.current_image['image'])

with c2:
    st.text("This image shows a number '5'.")
    st.slider("How convinced are you?", 0.0, 100.0, 50.0, step=0.1, format="%.1f%%", key="slider_value")
    cc1,cc2 = st.columns([1,2], vertical_alignment="bottom")
    with cc1:
        st.number_input("ML Steps", min_value=0, value=50, key="num_ml_steps")
    with cc2:
        st.button("Train", on_click=next_image, use_container_width=True)
    

