import streamlit as st
import pandas as pd
import pickle
import torch

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from src.gan import array_to_pil, Generator, Discriminator

st.set_page_config(page_title="Leaderboards", page_icon="🌟")
db_username = st.secrets.db_username
db_password = st.secrets.db_password

# @st.cache_resource
def load_pretrained_discriminator(model_path="models/pretrained_dicriminator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    discriminator = Discriminator().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator.eval()  # evaluation mode
    
    return discriminator, device

discriminator, device = load_pretrained_discriminator()
#if "discriminator" not in st.session_state:
st.session_state.discriminator = discriminator

@st.cache_data(ttl=60)
def evaluate_all_generators(_discriminator, num_samples=100):
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
        
        # Evaluate with _discriminator
        total_score = 0.0
        fake_images = []
        image_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate image
                z = torch.randn(1, generator.latent_dim)
                fake_image = generator(z)
                score = _discriminator(fake_image).item()

                fake_images.append(fake_image)
                image_scores.append(score)
                
                total_score += score
        results["Name"].append(row["name"])
        results["Score"].append(total_score)
        results["Images"].append(fake_images)
        results["I_Scores"].append(image_scores)

    paired_image_scores = list(zip(results["Images"], results["I_Scores"]))
    paired_image_scores.sort(key=lambda x: x[1], reverse=True)
    results["Images"], results["I_Scores"] = zip(*paired)
    results["Images"] = list(results["Images"])
    results["I_Scores"] = list(results["I_Scores"])

    return results

st.header("Leaderboards 🌟")
NUM_IMAGES = 5
with st.spinner():
    results = evaluate_all_generators(st.session_state.discriminator, 100)
    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    crowned = False
    for _, row in results_df.iterrows():
        disp_name = row["Name"]
        if not crowned:
            disp_name += " 👑"
            crowned = True
        with st.expander(f"[{row["Score"]*100:.5f}]  {disp_name}"):
            cs = st.columns(NUM_IMAGES)
            for j, c in enumerate(cs):
                with c:
                    st.image(array_to_pil(row["Images"][j],scale=3))
                    st.markdown(f"<p style='text-align: center;'>{row["I_Scores"][j]*100:.2f}</p>", unsafe_allow_html=True)
