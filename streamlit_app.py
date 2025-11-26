import streamlit as st
from torchvision import datasets, transforms
from torch import optim
import pandas as pd
import numpy as np

import sys
import os
import torch
import random
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.gan import (
    Generator, 
    Discriminator,
    generate_image, 
    get_random_real_image,
    train_gan_step,
    get_real_image_batch,
    array_to_pil
)
from pymongo import MongoClient
from pymongo.server_api import ServerApi

db_username = st.secrets.db_username
db_password = st.secrets.db_password

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
if "submitted" not in st.session_state:
    st.session_state.submitted = False

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

def on_submit_button():
    client = MongoClient(f"mongodb+srv://{db_username}:{db_password}@cluster0.5lnvrry.mongodb.net/?appName=Cluster0",
                     server_api=ServerApi('1'))
    st.session_state.submitted = True
    database = client.get_database("topic_sharing_demo")
    collection = database.get_collection("generators")
    if collection.find_one({"name": st.session_state.leaderboard_name}):
            st.error(f"Name '{st.session_state.leaderboard_name}' already exists! Please choose a different name.")
            return False
    state_dict = st.session_state.generator.state_dict()
    model_bytes = pickle.dumps(state_dict)

    collection.insert_one({
            "name": st.session_state.leaderboard_name,
            "model_data": model_bytes,  # Store as binary
            "training_steps": st.session_state.training_steps,
    })

    st.success(f"Generator submitted! '{st.session_state.leaderboard_name}'!")
    client.close()

def evaluate_all_generators(discriminator, num_samples=100):
    client = MongoClient(f"mongodb+srv://{db_username}:{db_password}@cluster0.5lnvrry.mongodb.net/?appName=Cluster0",
                    server_api=ServerApi('1'))
    db = client.get_database("topic_sharing_demo")
    collection = db.get_collection("generators")

    results = {
        "Name": [],
        "Score": [],
        "Images": [],
        "I_Scores": []
    }
    for row in collection.find():
        state_dict = pickle.loads(row["model_data"])
        generator = Generator()
        generator.load_state_dict(state_dict)
        generator.eval()
        
        # Evaluate with discriminator
        total_score = 0.0
        fake_images = []
        image_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate image
                z = torch.randn(1, generator.latent_dim)
                fake_image = generator(z)
                score = discriminator(fake_image).item()

                fake_images.append(fake_images)
                image_scores.append(score)
                
                total_score += score
        avg_score = total_score / num_samples
        results["Name"].append(row["name"])
        results["Score"].append(avg_score)
        results["Images"].append(fake_images)
        results["I_Scores"].append(image_scores)
    return results

st.divider()
ccc1, ccc2 = st.columns([1,1], vertical_alignment="bottom")
with ccc1:
    st.text_input("Name:", key="leaderboard_name")
with ccc2:
    # st.button("Submit Model to Leaderboards", disabled=st.session_state.submitted, on_click=on_submit_button)
    st.button("Submit to Leaderboards", on_click=on_submit_button)

st.info("You can only submit once!")
NUM_IMAGES = 10
if st.button("leaderboards"):
    with st.spinner():
        results = evaluate_all_generators(st.session_state.discriminator, 100)
        results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        for i, row in results_df.iterrows():
            # st.text(np.array(row["Images"][0]).shape)
            with st.expander(f"{row["Name"]}, Score: {row["Score"]}"):
                st.text("Images here")
                one_image = np.array(row["Images"][0])
                st.image(array_to_pil(one_image))
                #st.text(row["I_Scores"])
            #     cs = st.columns(NUM_IMAGES)
            #     for j, c in enumerate(cs):
            #         with c:
            #             pil_image = array_to_pil(row["Images"][j])
                        # st.image(pil_image)
                        


