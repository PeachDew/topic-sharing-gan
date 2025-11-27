import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gan import evaluate_all_generators, array_to_pil

st.set_page_config(page_title="Leaderboards", page_icon="🌟")

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