from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and prepare data
df = pd.read_csv("data/processed/clean_pcos_posts.csv")
corpus = df["clean_text"].tolist()

# Convert all posts to embeddings
print("📦 Generating embeddings for stories...")
corpus_embeddings = model.encode(corpus, show_progress_bar=True)

# 🔍 Function to match user input to similar stories
def find_similar_stories(user_input, top_n=3):
    query_embedding = model.encode([user_input])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]
    print(f"\n🧠 Top {top_n} matches for: \"{user_input}\"\n")
    for idx in top_indices:
        print(f"--- Match #{idx + 1} ---")
        print(df.iloc[idx]["clean_text"][:500] + "...")
        print(f"(Upvotes: {df.iloc[idx]['score']}, Link: https://reddit.com{df.iloc[idx]['permalink']})\n")

if __name__ == "__main__":
    test_query = "acne, weight gain, irregular periods"
    find_similar_stories(test_query)
