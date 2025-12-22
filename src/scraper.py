import praw
import pandas as pd
from datetime import datetime
import time

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_user_data(self, username, limit=100):
        """
        Fetches comments and basic metadata for a user.
        """
        try:
            user = self.reddit.redditor(username)
            # Force fetch to check if user exists/is not banned
            if hasattr(user, 'is_suspended') and user.is_suspended:
                return None
            
            comments_data = []
            for comment in user.comments.new(limit=limit):
                comments_data.append({
                    'text': comment.body,
                    'created_utc': comment.created_utc,
                    'score': comment.score,
                    'subreddit': comment.subreddit.display_name,
                    'is_submitter': comment.is_submitter
                })
            
            # Get account metadata
            account_data = {
                'username': username,
                'account_created_utc': user.created_utc,
                'link_karma': user.link_karma,
                'comment_karma': user.comment_karma,
                'verified': user.verified
            }
            
            return account_data, comments_data
            
        except Exception as e:
            print(f"Error fetching {username}: {e}")
            return None, None