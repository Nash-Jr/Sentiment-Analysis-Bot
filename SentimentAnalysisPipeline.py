from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pathlib import Path
import joblib
import os
import pandas as pd
import sys
import random
import shutil


def data_merger(neg_path, pos_path, mixed_path):
    # Create MixedData directory if it doesn't exist
    mixed_path.mkdir(parents=True, exist_ok=True)

    neg_files = list(neg_path.glob('*.txt'))
    pos_files = list(pos_path.glob('*.txt'))

    all_files = neg_files + pos_files
    random.shuffle(all_files)

    for file in all_files:
        dest_file = mixed_path / file.name
        if dest_file.exists():
            base_name = dest_file.stem
            extension = dest_file.suffix
            counter = 1
            while dest_file.exists():
                dest_file = mixed_path / f"{base_name}_{counter}{extension}"
                counter += 1

        shutil.copy2(file, dest_file)

    print(
        f"Combined {len(neg_files)} negative and {len(pos_files)} positive reviews into {mixed_path}")


# Set up paths
script_path = Path(__file__).resolve()
project_root = script_path.parent  # This should be the SentimentBot folder
sys.path.append(str(project_root))

sentiment_data_path = project_root / "SentimentData"
neg_reviews_path = sentiment_data_path / "train" / "neg"
pos_reviews_path = sentiment_data_path / "train" / "pos"
mixed_data_path = project_root / "MixedData"

# Merge the data
data_merger(neg_reviews_path, pos_reviews_path, mixed_data_path)
