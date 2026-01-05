import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import HybridBotDetector
import glob
import os

# --- 1. Custom Dataset Class ---
class RedditDataset(Dataset):
    def __init__(self, dataframe, tokenizer, meta_scaler=None, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Define which columns are metadata
        self.meta_cols = [
            'account_age_days', 'link_karma', 'comment_karma', 
            'is_gold', 'is_mod', 'verified', 
            'avg_comment_len', 'posting_variance', 'avg_score', 'unique_subreddits'
        ]
        
        # Normalize metadata (Neural networks like numbers between -1 and 1)
        if meta_scaler:
            self.meta_features = meta_scaler.transform(self.data[self.meta_cols].fillna(0))
        else:
            self.meta_features = self.data[self.meta_cols].fillna(0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get text
        text = str(self.data.iloc[index]['combined_text'])
        label = self.data.iloc[index]['label']
        
        # Tokenize text for BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'meta_features': torch.tensor(self.meta_features[index], dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --- 2. Main Training Loop ---
def train():
    # Setup Device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (Find the latest CSV created by your pipeline)
    list_of_files = glob.glob('training_data_*.csv') 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Split Data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare Scaler for metadata
    scaler = StandardScaler()
    meta_cols = [
        'account_age_days', 'link_karma', 'comment_karma', 
        'is_gold', 'is_mod', 'verified', 
        'avg_comment_len', 'posting_variance', 'avg_score', 'unique_subreddits'
    ]
    scaler.fit(train_df[meta_cols].fillna(0))

    # Prepare Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Create DataLoaders
    train_dataset = RedditDataset(train_df, tokenizer, meta_scaler=scaler)
    val_dataset = RedditDataset(val_df, tokenizer, meta_scaler=scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize Model
    model = HybridBotDetector(meta_input_dim=len(meta_cols))
    model = model.to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss() # Binary Cross Entropy for 0 or 1

    # Training Loop
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            meta_features = batch['meta_features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, meta_features)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Avg Loss: {total_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), "bot_detector_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    train()