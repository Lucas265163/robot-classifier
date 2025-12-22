import numpy as np
import pandas as pd
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_metadata(self, account_data, comments_data):
        """
        Derives behavioral features from raw account and comment data.
        Returns a dictionary of numerical features.
        """
        if not comments_data:
            return None

        df = pd.DataFrame(comments_data)
        
        # 1. Posting Speed (Average seconds between comments)
        if len(df) > 1:
            timestamps = df['created_utc'].sort_values().values
            diffs = np.diff(timestamps)
            avg_inter_comment_gap = np.mean(diffs)
            std_inter_comment_gap = np.std(diffs)
        else:
            avg_inter_comment_gap = 0
            std_inter_comment_gap = 0

        # 2. Text Repetition (Are they spamming the same phrase?)
        unique_comments = df['text'].nunique()
        total_comments = len(df)
        repetition_ratio = 1.0 - (unique_comments / total_comments)

        # 3. Account Age (in days)
        account_age_days = (datetime.now().timestamp() - account_data['account_created_utc']) / 86400

        # 4. Karma Ratios
        total_karma = account_data['link_karma'] + account_data['comment_karma']
        karma_ratio = account_data['comment_karma'] / (total_karma + 1) # Avoid div by zero

        return {
            'avg_gap_seconds': avg_inter_comment_gap,
            'std_gap_seconds': std_inter_comment_gap,
            'repetition_ratio': repetition_ratio,
            'account_age_days': account_age_days,
            'karma_ratio': karma_ratio,
            'comment_count': total_comments
        }