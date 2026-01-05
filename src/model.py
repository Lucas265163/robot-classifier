import torch
import torch.nn as nn
from transformers import DistilBertModel

class HybridBotDetector(nn.Module):
    def __init__(self, meta_input_dim):
        super(HybridBotDetector, self).__init__()
        
        # 1. Text Branch: DistilBERT (Pre-trained)
        # We use 'distilbert-base-uncased' which is light and fast.
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 2. Metadata Branch: Simple Neural Network
        # This processes features like Account Age, Karma, etc.
        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)  # Helps with training stability
        )
        
        # 3. Fusion Head: Combines both
        # DistilBERT outputs 768 dims, Meta branch outputs 16 dims
        self.classifier = nn.Sequential(
            nn.Linear(768 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevents overfitting
            nn.Linear(64, 1),
            nn.Sigmoid()     # Outputs probability between 0 and 1
        )

    def forward(self, input_ids, attention_mask, meta_features):
        # A. Process Text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # We take the embedding of the [CLS] token (the first token) as the summary of the text
        text_emb = bert_output.last_hidden_state[:, 0, :]
        
        # B. Process Metadata
        meta_emb = self.meta_net(meta_features)
        
        # C. Combine (Concatenate)
        combined = torch.cat((text_emb, meta_emb), dim=1)
        
        # D. Predict
        return self.classifier(combined)