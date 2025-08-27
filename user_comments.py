import os
import pickle
from collections import defaultdict
from fileStreams import getFileJsonStream

def simple_tokenize(text):
    return text.lower().split()

def collect_comments_by_user(zst_path):
    user_comments = defaultdict(list)
    with open(zst_path, "rb") as f:
        jsonStream = getFileJsonStream(zst_path, f)
        if jsonStream is None:
            raise ValueError(f"Cannot read file: {zst_path}")
        for row in jsonStream:
            if "author" not in row or "body" not in row:
                continue
            author = row["author"]
            text = row["body"]
            tokens = simple_tokenize(text)
            user_comments[author].append(tokens)
    return dict(user_comments)

if __name__ == "__main__":

    zst_path = "data/democrats_comments.zst"
    output_path = "output/user_comments/democrats.pkl"
    os.makedirs("output/user_comments", exist_ok=True)

    print(f"Handling {zst_path} ...")
    user_comments = collect_comments_by_user(zst_path)
    print(f"Collected {len(user_comments)} users, total comments: {sum(len(coms) for coms in user_comments.values())}")

    with open(output_path, "wb") as f:
        pickle.dump(user_comments, f)
    print(f"Saved to {output_path}")

    # Print a sample user and their comments
    sample_user = next(iter(user_comments))
    print(f"Sample user: {sample_user}, Comments: {len(user_comments[sample_user])}")
    print("First two tokenized comments:", user_comments[sample_user][:2])