import torch
import os
import numpy as np
from dotenv import load_dotenv
from transformers import DistilBertTokenizer
from model import HybridBotDetector
from data_pipline import RedditDataPipeline, logging # Reuse your pipeline!

# Load Env
load_dotenv()

def predict_user(username):
    # 1. Setup
    device = torch.device("cpu") # Laptops usually use CPU for single predictions
    print(f"Analyzing user: {username}...")

    # 2. Fetch Live Data (Using your pipeline)
    pipeline = RedditDataPipeline(
        os.getenv("REDDIT_CLIENT_ID"), 
        os.getenv("REDDIT_CLIENT_SECRET"), 
        os.getenv("REDDIT_USER_AGENT")
    )
    
    # We pass '0' as label because we don't know it yet, we just want the features
    user_data = pipeline.process_user(username, label=0)
    
    if not user_data:
        print("User not found or suspended.")
        return

    # 3. Prepare Data for Model
    # A. Text
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer.encode_plus(
        user_data['combined_text'],
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # B. Metadata (Must match the order in train.py!)
    meta_cols = [
        'account_age_days', 'link_karma', 'comment_karma', 
        'is_gold', 'is_mod', 'verified', 
        'avg_comment_len', 'posting_variance', 'avg_score', 'unique_subreddits'
    ]
    # In a real app, you would load the 'scaler' from training. 
    # For now, we use raw values or manual normalization.
    meta_values = [user_data[col] for col in meta_cols]
    meta_tensor = torch.tensor([meta_values], dtype=torch.float)

    # 4. Load Model
    model = HybridBotDetector(meta_input_dim=len(meta_cols))
    try:
        model.load_state_dict(torch.load("bot_detector_model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: Model file 'bot_detector_model.pth' not found. Run train.py first!")
        return

    model.eval() # Set to evaluation mode

    # 5. Predict
    with torch.no_grad():
        output = model(
            encoding['input_ids'], 
            encoding['attention_mask'], 
            meta_tensor
        )
        probability = output.item()

    # 6. Result
    print(f"\n--- REPORT FOR {username} ---")
    print(f"Bot Probability: {probability:.4f} ({probability*100:.1f}%)")
    if probability > 0.5:
        print("Verdict: 🤖 BOT DETECTED")
    else:
        print("Verdict: 👤 HUMAN DETECTED")

if __name__ == "__main__":
    # Test on a known bot and a known human
    predict_user("AutoModerator") 
    # predict_user("YourUsernameHere") 