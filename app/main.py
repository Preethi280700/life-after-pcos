st.set_page_config(page_title="Life After PCOS", page_icon="💡")

with st.sidebar:
    st.title("💡 Life After PCOS")
    st.markdown("👩‍⚕️ Personalized PCOS recovery guidance based on real journeys.")
    st.markdown("🔍 Powered by NLP + real Reddit recovery stories.")
    st.markdown("---")
    st.markdown("📌 Built by **Preethi Bommineni**")

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/processed/clean_pcos_posts.csv")
corpus = df["clean_text"].tolist()
corpus_embeddings = model.encode(corpus, show_progress_bar=False)

# App UI
st.title("💡 Life After PCOS")
st.markdown("Enter your symptoms or experience to see what worked for others.")

st.markdown("### 🩺 Describe your PCOS symptoms or situation:")

with st.form(key="symptom_form"):
    user_input = st.text_area(" ", placeholder="E.g. acne, hair loss, missed periods, fatigue")
    submit_button = st.form_submit_button("🔍 Find Similar Stories")

if submit_button and user_input.strip():
    with st.spinner("Matching your story with others..."):
        query_embedding = model.encode([user_input])
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]

        st.success("Here are similar stories from real users:")

        for idx in top_indices:
            st.markdown("### ✨ Story Match")
            st.write(df.iloc[idx]["clean_text"][:500] + "...")
            st.markdown(f"[🔗 View on Reddit](https://reddit.com{df.iloc[idx]['permalink']})")
            st.markdown(f"👍 Upvotes: {df.iloc[idx]['score']}")
            st.markdown("---")
