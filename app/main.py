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
    st.markdown("Personalized PCOS recovery guidance based on real experiences.")
    st.markdown("Powered by NLP + real Reddit & user-submitted stories.")
    st.markdown("---")
    st.markdown("🔗 Built by **Preethi Bommineni**")

# Session state for form navigation
if "step" not in st.session_state:
    st.session_state.step = 1

# Load model and datasets
model = SentenceTransformer("all-MiniLM-L6-v2")

df_main = pd.read_csv("data/processed/clean_pcos_posts.csv")

# Load user-contributed stories if available
try:
    df_user = pd.read_csv("data/processed/user_stories.csv")
    df_user["clean_text"] = df_user["story"]
    df = pd.concat([df_main[["clean_text", "permalink", "score"]], df_user[["clean_text"]]], ignore_index=True)
except FileNotFoundError:
    df = df_main[["clean_text", "permalink", "score"]]

# Vectorize all stories
corpus = df["clean_text"].tolist()
corpus_embeddings = model.encode(corpus, show_progress_bar=False)

# -------------------
# STEP 1: Profile Form
# -------------------
if st.session_state.step == 1:
    st.title("🧬 Life After PCOS – Step 1: Tell Us About You")
    st.markdown("This helps us understand your background and symptoms so we can offer personalized guidance.")

    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Name")
            dob = st.date_input("Date of Birth", min_value=datetime(1960, 1, 1), max_value=datetime.now())

        with col2:
            gender = st.radio("Gender", ["Female", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])

        story = st.text_area("🩺 Briefly describe your symptoms or experience with PCOS")

        next_button = st.form_submit_button("➡️ Continue")

        if next_button:
            if name.strip() == "" or story.strip() == "":
                st.warning("Please fill in your name and experience before continuing.")
            else:
                st.session_state["name"] = name
                st.session_state["dob"] = dob
                st.session_state["gender"] = gender
                st.session_state["marital_status"] = marital_status
                st.session_state["story"] = story
                st.session_state.step = 2

# -------------------
# STEP 2: Question + Matcher
# -------------------
elif st.session_state.step == 2:
    st.title("💬 Step 2: What’s On Your Mind?")
    st.markdown("We’ll match your story with real experiences that relate to your situation.")

    with st.form("help_form"):
        question = st.text_area("What questions or concerns would you like support on?")
        submit_button = st.form_submit_button("🔍 Show Matches")

        if submit_button and question.strip():
            st.session_state["question"] = question

            # Save user story to dataset
            new_entry = pd.DataFrame([{
                "name": st.session_state["name"],
                "dob": st.session_state["dob"],
                "gender": st.session_state["gender"],
                "marital_status": st.session_state["marital_status"],
                "story": st.session_state["story"],
                "question": st.session_state["question"]
            }])

            save_path = "data/processed/user_stories.csv"
            try:
                existing = pd.read_csv(save_path)
                updated = pd.concat([existing, new_entry], ignore_index=True)
            except FileNotFoundError:
                updated = new_entry
            updated.to_csv(save_path, index=False)

            # Show results
            st.success(f"Thanks {st.session_state['name']}! Here are similar stories we found:")

            query_embedding = model.encode([st.session_state["story"]])
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            top_indices = similarities.argsort()[-3:][::-1]

            for idx in top_indices:
                st.markdown("### ✨ Story Match")
                st.write(df.iloc[idx]["clean_text"][:500] + "...")
                if "permalink" in df.columns:
                    st.markdown(f"[🔗 View on Reddit](https://reddit.com{df.iloc[idx]['permalink']})")
                if "score" in df.columns:
                    st.markdown(f"👍 Upvotes: {df.iloc[idx]['score']}")
                st.markdown("---")
