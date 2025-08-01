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

user_input = st.text_area("💬 Describe your PCOS symptoms or situation:", "")

if st.button("🔍 Find Similar Stories") and user_input.strip() != "":
    query_embedding = model.encode([user_input])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]

    st.subheader("🔗 Top Recovery Stories")
    for idx in top_indices:
        st.markdown(f"**Reddit Post**: [Link](https://reddit.com{df.iloc[idx]['permalink']})")
        st.markdown(f"**Upvotes**: {df.iloc[idx]['score']}")
        st.write(df.iloc[idx]["clean_text"][:500] + "...")
        st.markdown("---")
