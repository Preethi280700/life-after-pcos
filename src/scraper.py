import praw
import pandas as pd

# TODO: Replace these with your own credentials
client_id = "ZyGXvAB9gViE4uu4c9sMKA"
client_secret = "so8niF9GLG0obZsqK65PzXk1-KnLEQ"
user_agent = "life-after-pcos-script by Beneficial_Prior81"

# Connect to Reddit
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

def scrape_pcos_posts(subreddit_name="PCOS", limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.search("pcos recovery", sort="top", time_filter="all", limit=limit):
        posts_data.append({
            "title": post.title,
            "selftext": post.selftext,
            "url": post.url,
            "score": post.score,
            "created_utc": post.created_utc,
            "permalink": post.permalink
        })

    return posts_data

if __name__ == "__main__":
    posts = scrape_pcos_posts(limit=200)
    df = pd.DataFrame(posts)
    df.to_csv("data/raw/reddit_pcos_stories.csv", index=False)
    print(f"Saved {len(df)} posts to CSV ✅")
