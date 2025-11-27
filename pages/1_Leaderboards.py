import streamlit as st
import pandas as pd
import pickle
import torch


from pymongo import MongoClient
from pymongo.server_api import ServerApi
from src.gan import array_to_pil, Generator

st.set_page_config(page_title="Leaderboards", page_icon="🌟")
db_username = st.secrets.db_username
db_password = st.secrets.db_password

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

                fake_images.append(fake_image)
                image_scores.append(score)
                
                total_score += score
        results["Name"].append(row["name"])
        results["Score"].append(total_score)
        results["Images"].append(fake_images)
        results["I_Scores"].append(image_scores)
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
        with st.expander(f"[{row["Score"]:.1f}]  {disp_name}"):
            cs = st.columns(NUM_IMAGES)
            for j, c in enumerate(cs):
                with c:
                    st.image(array_to_pil(row["Images"][j],scale=3))
                    st.markdown(f"<p style='text-align: center;'>{row["I_Scores"][j]*100:.1f}</p>", unsafe_allow_html=True)
