import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Set up Streamlit page
st.set_page_config(page_title="Life After PCOS", page_icon="💡")

# Sidebar navigation + credits
with st.sidebar:
    st.title("💡 Life After PCOS")
    st.markdown("Personalized PCOS recovery guidance based on real experiences.")
    st.markdown("Powered by NLP + real Reddit & user-submitted stories.")
    st.markdown("---")
    st.markdown("🔗 Built by **Preethi Bommineni**")
    if st.button("📚 View Anonymous Stories"):
        st.session_state.step = 99

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
df_main = pd.read_csv("data/processed/clean_pcos_posts.csv")

# Load user-contributed stories if available
try:
    df_user = pd.read_csv("data/processed/user_stories.csv")
    df_user["clean_text"] = df_user["story"]
    df = pd.concat([df_main[["clean_text", "permalink", "score"]], df_user[["clean_text"]]], ignore_index=True)
except FileNotFoundError:
    df = df_main[["clean_text", "permalink", "score"]]

# Generate embeddings
corpus = df["clean_text"].tolist()
corpus_embeddings = model.encode(corpus, show_progress_bar=False)

# -------------------------------
# STEP 1: Basic Profile Collection
# -------------------------------
if st.session_state.step == 1:
    st.title("🧬 Life After PCOS – Step 1: Tell Us About You")
    st.markdown("This helps us personalize the experience for you.")

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name")
            dob = st.date_input("Date of Birth", min_value=datetime(1960, 1, 1), max_value=datetime.now())
        with col2:
            gender = st.radio("Gender", ["Female", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])

        story = st.text_area("🩺 Describe your PCOS symptoms or experience")

        next_button = st.form_submit_button("➡️ Continue")

        if next_button:
            if name.strip() == "" or story.strip() == "":
                st.warning("Please fill in your name and story before continuing.")
            else:
                st.session_state["name"] = name
                st.session_state["dob"] = dob
                st.session_state["gender"] = gender
                st.session_state["marital_status"] = marital_status
                st.session_state["story"] = story
                st.session_state.step = 2

# -------------------------------
# STEP 2: Question + Matching
# -------------------------------
elif st.session_state.step == 2:
    st.title("💬 Step 2: What’s On Your Mind?")
    st.markdown("We’ll match your story with similar recovery experiences.")

    with st.form("help_form"):
        question = st.text_area("What questions or concerns would you like support on?")
        submit_button = st.form_submit_button("🔍 Show Matches")

        if submit_button and question.strip():
            st.session_state["question"] = question

            # Save user story
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

            # Matching
            st.success(f"Thanks {st.session_state['name']}! Here are similar recovery stories:")

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

            # Thank-you + Download
            st.success("🎉 You're not alone. We're with you every step of the way.")

            user_data = f"""
Name: {st.session_state['name']}
DOB: {st.session_state['dob']}
Gender: {st.session_state['gender']}
Marital Status: {st.session_state['marital_status']}

Story:
{st.session_state['story']}

Question:
{st.session_state['question']}
"""

            st.download_button(
                label="📄 Download My Submission",
                data=user_data,
                file_name=f"{st.session_state['name'].replace(' ', '_')}_pcos_story.txt",
                mime='text/plain'
            )

# -------------------------------
# STEP 99: Anonymous Story Viewer
# -------------------------------
elif st.session_state.step == 99:
    st.title("📚 Anonymous PCOS Stories")

    try:
        stories = pd.read_csv("data/processed/user_stories.csv")
        for i, row in stories.iterrows():
            st.markdown(f"### ✍️ Story #{i+1}")
            st.write(row["story"])
            st.markdown(f"**Question Asked:** {row['question']}")
            st.markdown("---")
    except FileNotFoundError:
        st.warning("No user-submitted stories available yet.")
