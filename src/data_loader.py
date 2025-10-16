"""
Data loader module for IMDB Sentiment Analysis.
Loads cleaned data and prepares it for model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH_CLEANED, VALIDATION_SPLIT, RANDOM_SEED


def check_data_exists():
    """
    Check if the cleaned dataset file exists.

    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.exists(DATA_PATH_CLEANED):
        print(f"❌ Error: Cleaned dataset not found at {DATA_PATH_CLEANED}")
        print("Please run data cleaning first: python src/data_clean.py")
        return False
    print(f"✅ Cleaned dataset found: {DATA_PATH_CLEANED}")
    return True


def load_data():
    """
    Load the cleaned IMDB dataset from CSV file.

    Returns:
        pd.DataFrame: Dataframe with 'review' and 'sentiment' columns
        None: If file doesn't exist or loading fails
    """
    # Check if file exists
    if not check_data_exists():
        return None

    try:
        print("\n" + "=" * 70)
        print("LOADING CLEANED DATA")
        print("=" * 70 + "\n")

        print(f"📖 Reading: {DATA_PATH_CLEANED}")

        # Load cleaned CSV
        df = pd.read_csv(DATA_PATH_CLEANED, encoding='utf-8')

        # Verify columns
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            print(f"❌ Expected columns 'review' and 'sentiment', got: {df.columns.tolist()}")
            return None

        print(f"✅ Data loaded successfully!")
        print(f"   Total samples: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def explore_data(df):
    """
    Print basic information about the dataset.

    Args:
        df (pd.DataFrame): The dataset to explore
    """
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70 + "\n")

    # Basic info
    print(f"📊 Shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")

    # Sentiment distribution
    print("\n📊 Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count:,} ({percentage:.2f}%)")

    # Sample reviews
    print("\n📋 Sample Reviews:")
    for i, row in df.head(3).iterrows():
        review_preview = row['review'][:100] + "..." if len(row['review']) > 100 else row['review']
        print(f"   [{row['sentiment']}] {review_preview}")

    print("\n" + "=" * 70 + "\n")


def prepare_data(df):
    """
    Prepare data for training: extract features and convert labels to binary.

    Args:
        df (pd.DataFrame): The dataset

    Returns:
        tuple: (reviews, labels)
               - reviews: array of review texts
               - labels: binary array (1=positive, 0=negative)
    """
    print("=" * 70)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 70 + "\n")

    # Extract reviews (feature/input data)
    reviews = df['review'].values

    # Convert sentiment strings to binary labels
    # positive -> 1, negative -> 0
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values

    # Calculate statistics
    positive_count = np.sum(labels)
    negative_count = len(labels) - positive_count
    positive_pct = (positive_count / len(labels)) * 100
    negative_pct = (negative_count / len(labels)) * 100

    print(f"✅ Data prepared:")
    print(f"   Total reviews: {len(reviews):,}")
    print(f"   Positive (1): {positive_count:,} ({positive_pct:.2f}%)")
    print(f"   Negative (0): {negative_count:,} ({negative_pct:.2f}%)")

    return reviews, labels


def split_data(reviews, labels):
    """
    Split data into training and validation sets.
    Uses stratified split to maintain class balance in both sets.

    Args:
        reviews (np.array): Review texts
        labels (np.array): Sentiment labels (binary)

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    print("\n" + "=" * 70)
    print("SPLITTING DATA (TRAIN/VALIDATION)")
    print("=" * 70 + "\n")

    X_train, X_val, y_train, y_val = train_test_split(
        reviews,
        labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels  # Keep same positive/negative ratio in both sets
    )

    # Calculate statistics
    train_pct = (1 - VALIDATION_SPLIT) * 100
    val_pct = VALIDATION_SPLIT * 100

    train_pos = np.sum(y_train)
    train_neg = len(y_train) - train_pos
    val_pos = np.sum(y_val)
    val_neg = len(y_val) - val_pos

    print(f"✅ Data split complete:")
    print(f"\n   📚 Training Set: {len(X_train):,} samples ({train_pct:.0f}%)")
    print(f"      - Positive: {train_pos:,} ({(train_pos / len(y_train)) * 100:.2f}%)")
    print(f"      - Negative: {train_neg:,} ({(train_neg / len(y_train)) * 100:.2f}%)")

    print(f"\n   🧪 Validation Set: {len(X_val):,} samples ({val_pct:.0f}%)")
    print(f"      - Positive: {val_pos:,} ({(val_pos / len(y_val)) * 100:.2f}%)")
    print(f"      - Negative: {val_neg:,} ({(val_neg / len(y_val)) * 100:.2f}%)")

    return X_train, X_val, y_train, y_val


def load_and_prepare_data(verbose=True):
    """
    Main pipeline function: loads cleaned data and prepares it for training.

    Pipeline steps:
    1. Load cleaned data
    2. (Optional) Explore data
    3. Prepare features and labels
    4. Split into train/validation sets

    Args:
        verbose (bool): If True, print data exploration details

    Returns:
        tuple: (X_train, X_val, y_train, y_val) or None if any step fails
    """
    print("\n" + "=" * 70)
    print("🚀 DATA LOADING & PREPARATION PIPELINE")
    print("=" * 70)

    # Step 1: Load cleaned data
    df = load_data()
    if df is None:
        return None

    # Step 2: Explore data (optional)
    if verbose:
        explore_data(df)

    # Step 3: Prepare features and labels
    reviews, labels = prepare_data(df)

    # Step 4: Split into train/validation sets
    X_train, X_val, y_train, y_val = split_data(reviews, labels)

    print("\n" + "=" * 70)
    print("🎉 DATA PREPARATION COMPLETE!")
    print("=" * 70 + "\n")

    return X_train, X_val, y_train, y_val


# Test the module if run directly
if __name__ == "__main__":
    """
    Test block - runs only when this file is executed directly.
    Usage: python src/data_loader.py
    """
    print("\n" + "=" * 70)
    print("TESTING DATA_LOADER MODULE")
    print("=" * 70)

    result = load_and_prepare_data(verbose=True)

    if result is not None:
        X_train, X_val, y_train, y_val = result
        print("\n✅ Module test successful!")
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Validation set: {len(X_val):,} samples")
        print("\n" + "=" * 70 + "\n")
    else:
        print("\n❌ Module test failed!")
        print("=" * 70 + "\n")

