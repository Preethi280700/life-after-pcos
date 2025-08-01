import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Set page
st.set_page_config(page_title="Life After PCOS", page_icon="💡")

# Sidebar
with st.sidebar:
    st.title("💡 Life After PCOS")
    st.markdown("👩‍⚕️ Personalized PCOS recovery guidance based on real journeys.")
    st.markdown("🔍 Powered by NLP + real Reddit recovery stories.")
    st.markdown("---")
    st.markdown("📌 Built by **Preethi Bommineni**")

# Session state
if "step" not in st.session_state:
    st.session_state.step = 1

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/processed/clean_pcos_posts.csv")
corpus = df["clean_text"].tolist()
corpus_embeddings = model.encode(corpus, show_progress_bar=False)

# 🔹 STEP 1: Basic Info
if st.session_state.step == 1:
    st.title("🧬 Life After PCOS – Step 1: Your Profile")

    with st.form("profile_form"):
        name = st.text_input("Your Name")
        dob = st.date_input("Date of Birth")
        gender = st.radio("Gender", ["Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        story = st.text_area("Describe your PCOS symptoms or experience")

        next_button = st.form_submit_button("➡️ Next")

        if next_button and name and story:
            st.session_state["name"] = name
            st.session_state["dob"] = dob
            st.session_state["gender"] = gender
            st.session_state["marital_status"] = marital_status
            st.session_state["story"] = story
            st.session_state.step = 2

# 🔹 STEP 2: Questions + Matching
elif st.session_state.step == 2:
    st.title("🩺 Step 2: How Can We Help You?")

    with st.form("help_form"):
        question = st.text_area("What questions or concerns do you have?")
        submit_button = st.form_submit_button("🔍 Show Matches")

        if submit_button and question:
            st.session_state["question"] = question

            st.subheader(f"Hi {st.session_state['name']} 👋")
            st.markdown(f"**Your concern:** {question}")
            st.markdown(f"**Your story:** {st.session_state['story']}")

            with st.spinner("Matching your story with others..."):
                query_embedding = model.encode([st.session_state['story']])
                similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
                top_indices = similarities.argsort()[-3:][::-1]

                st.success("Here are similar recovery stories:")

                for idx in top_indices:
                    st.markdown("### ✨ Story Match")
                    st.write(df.iloc[idx]["clean_text"][:500] + "...")
                    st.markdown(f"[🔗 View on Reddit](https://reddit.com{df.iloc[idx]['permalink']})")
                    st.markdown(f"👍 Upvotes: {df.iloc[idx]['score']}")
                    st.markdown("---")
