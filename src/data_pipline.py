import praw
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)

class RedditDataPipeline:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize the Reddit API connection.
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logging.info("Connected to Reddit API.")

    def get_user_metadata(self, username: str) -> Dict[str, Union[float, int]]:
        """
        Fetch raw metadata for a specific user.
        Features: Account Age, Karma, Verification Status.
        """
        try:
            user = self.reddit.redditor(username)
            # Force fetch data
            _ = user.id 
            
            created_utc = user.created_utc
            account_age_days = (time.time() - created_utc) / 86400
            
            return {
                "username": username,
                "account_age_days": account_age_days,
                "link_karma": user.link_karma,
                "comment_karma": user.comment_karma,
                "is_gold": int(user.is_gold),
                "is_mod": int(user.is_mod),
                "verified": int(user.has_verified_email) if hasattr(user, 'has_verified_email') else 0,
                "fetch_status": "success"
            }
        except Exception as e:
            logging.error(f"Error fetching metadata for {username}: {e}")
            return {"username": username, "fetch_status": "failed"}

    def get_user_comments(self, username: str, limit: int = 100) -> List[Dict]:
        """
        Fetch the last N comments to analyze text and posting behavior.
        """
        comments_data = []
        try:
            user = self.reddit.redditor(username)
            for comment in user.comments.new(limit=limit):
                comments_data.append({
                    "username": username,
                    "text": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "subreddit": comment.subreddit.display_name
                })
        except Exception as e:
            logging.warning(f"Could not fetch comments for {username}: {e}")
        
        return comments_data

    def engineer_behavioral_features(self, comments_data: List[Dict]) -> Dict[str, float]:
        """
        CRITICAL GRAD SCHOOL STEP: Feature Engineering.
        Derive complex signals from raw logs.
        """
        if not comments_data:
            return {
                "avg_comment_len": 0,
                "posting_variance": 0,
                "avg_score": 0,
                "unique_subreddits": 0
            }

        df = pd.DataFrame(comments_data)
        
        # Feature 1: Average Comment Length (Bots often have fixed templates)
        df['length'] = df['text'].apply(len)
        avg_len = df['length'].mean()

        # Feature 2: Posting Variance (Bots are clockwork, Humans are erratic)
        # Calculate time difference between consecutive posts
        df = df.sort_values('created_utc')
        df['time_diff'] = df['created_utc'].diff()
        posting_variance = df['time_diff'].std()
        if pd.isna(posting_variance):
            posting_variance = 0

        # Feature 3: Subreddit Diversity
        unique_subs = df['subreddit'].nunique()

        return {
            "avg_comment_len": avg_len,
            "posting_variance": posting_variance,
            "avg_score": df['score'].mean(),
            "unique_subreddits": unique_subs
        }

    def process_user(self, username: str, label: int) -> Dict:
        """
        Master function to build a single training example (Text + Metadata).
        Label: 1 for Bot, 0 for Human.
        """
        # 1. Get Static Metadata
        meta = self.get_user_metadata(username)
        if meta["fetch_status"] == "failed":
            return None

        # 2. Get Dynamic Comments
        comments = self.get_user_comments(username, limit=50)
        
        # 3. Engineer Features from Comments
        behavioral = self.engineer_behavioral_features(comments)
        
        # 4. Concatenate all text for the BERT model
        combined_text = " [SEP] ".join([c['text'] for c in comments[:10]]) # Only take last 10 for speed

        # Merge everything
        return {
            **meta,
            **behavioral,
            "combined_text": combined_text,
            "label": label
        }

    def build_dataset(self, user_list: List[tuple], max_workers: int = 4):
        """
        Uses ThreadPoolExecutor to fetch data in parallel (High Performance).
        user_list: list of (username, label) tuples.
        """
        results = []
        logging.info(f"Starting data collection for {len(user_list)} users...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_user = {executor.submit(self.process_user, user, label): user for user, label in user_list}
            
            for future in future_to_user:
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as exc:
                    logging.error(f"Generated an exception: {exc}")

        # Save to CSV
        df = pd.DataFrame(results)
        output_file = f"training_data_{int(time.time())}.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Dataset saved to {output_file} with {len(df)} records.")
        return df

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # You must fill these in from https://www.reddit.com/prefs/apps
    CLIENT_ID = "YOUR_CLIENT_ID"
    CLIENT_SECRET = "YOUR_CLIENT_SECRET"
    USER_AGENT = "python:bot_detector:v1.0 (by /u/Lucas265163)"

    pipeline = RedditDataPipeline(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    # --- DEFINE YOUR TARGETS ---
    # In a real scenario, you load these from a file.
    # Format: (Username, 1=Bot/0=Human)
    targets = [
        ("AutoModerator", 1),
        ("RemindMeBot", 1),
        ("Wikipedia", 1),
        ("spez", 0),  # Reddit CEO
        ("BarackObama", 0)
    ]

    df = pipeline.build_dataset(targets)
    print(df.head())