import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # remove links
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text) # remove emojis/symbols
    text = re.sub(r"\s+", " ", text).strip()       # extra spaces
    return text

def filter_and_clean(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["selftext"], inplace=True)
    
    # Keep only long-form, meaningful posts
    df["text_length"] = df["selftext"].apply(lambda x: len(str(x)))
    df = df[df["text_length"] > 300]  # keep posts with >300 characters

    # Clean the post content
    df["clean_text"] = df["selftext"].apply(clean_text)
    
    df[["title", "clean_text", "permalink", "score"]].to_csv(output_path, index=False)
    print(f"✅ Cleaned and saved {len(df)} posts to {output_path}")

if __name__ == "__main__":
    filter_and_clean("data/raw/reddit_pcos_stories.csv", "data/processed/clean_pcos_posts.csv")
